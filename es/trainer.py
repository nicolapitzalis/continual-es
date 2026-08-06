import ray
import torch
import random
import numpy as np
from dataclasses import replace

from es.checkpoint import (
    load_checkpoint,
    prepare_continual_tasks,
    restore_policy,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from es.config import TrainingConfig
from es.envs import extract_envs_info, make_env
from es.evaluation import evaluate_tasks
from es.horizon import HorizonManager
from es.metrics import CsvLogger, build_run_name, log_fields
from es.noise import NoiseGenerator
from es.normalization import ObservationNormalizer
from es.policy import Policy, get_flat_params, set_flat_params
from es.replay import compute_replay_gradient
from es.sampling import collect_rollouts
from es.updates import apply_es_gradient, compute_gradient
from es.worker import ESWorker


def _add_evaluations_to_log_row(
    log_row,
    *,
    policy,
    environments,
    env_names,
    current_task_id,
    multi_task,
    config,
    episode_seeds,
    normalizer,
):
    if multi_task:
        evaluation_task_ids = list(range(len(env_names)))
    else:
        evaluation_task_ids = list(range(current_task_id + 1))

    evaluation_rewards = evaluate_tasks(
        policy,
        environments,
        evaluation_task_ids,
        action_bins=config.action_bins,
        episode_seeds=episode_seeds,
        normalizer=normalizer,
        obs_clip=config.obs_clip,
    )
    for evaluation_task_id, evaluation_reward in evaluation_rewards.items():
        task_name = env_names[evaluation_task_id]
        if multi_task:
            log_row[f"{task_name}_eval_curr_policy"] = evaluation_reward
            label = "current policy on"
        elif evaluation_task_id == current_task_id:
            log_row["eval_curr_policy"] = evaluation_reward
            label = "current policy on"
        else:
            log_row[f"{task_name}_avg_reward"] = evaluation_reward
            label = "old"
        print(
            f"Eval {label} env {task_name}: "
            f"avg reward = {evaluation_reward:.2f}"
        )

    if not config.fwt:
        return

    future_task_ids = list(range(current_task_id + 1, len(env_names)))
    missing_task_ids = [
        task_id
        for task_id in future_task_ids
        if task_id not in evaluation_rewards
    ]
    fwt_rewards = {
        task_id: evaluation_rewards[task_id]
        for task_id in future_task_ids
        if task_id in evaluation_rewards
    }
    if missing_task_ids:
        fwt_rewards.update(
            evaluate_tasks(
                policy,
                environments,
                missing_task_ids,
                action_bins=config.action_bins,
                episode_seeds=episode_seeds,
                normalizer=normalizer,
                obs_clip=config.obs_clip,
            )
        )
    for future_task_id, fwt_reward in fwt_rewards.items():
        task_name = env_names[future_task_id]
        log_row[f"{task_name}_fwt_before"] = fwt_reward
        print(
            f"Eval next env {task_name} for FWT: "
            f"avg reward = {fwt_reward:.2f}"
        )


def train(envs, config=None, checkpoint=None, ray_address=None):
    config = TrainingConfig() if config is None else config
    env_names = list(envs)
    if not env_names:
        raise ValueError("At least one environment must be specified")

    checkpoint_data = None
    if checkpoint:
        checkpoint_data = load_checkpoint(checkpoint)
        validate_checkpoint_compatibility(checkpoint_data, config)
        env_names, current_task_id = prepare_continual_tasks(
            checkpoint_data,
            env_names,
        )
        config = replace(
            config,
            shared_output=checkpoint_data["shared_output"],
        )
        multi_task = False
    else:
        current_task_id = 0
        multi_task = len(env_names) > 1

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    environments = [make_env(env_name) for env_name in env_names]
    input_dims, output_dims, output_activations = extract_envs_info(
        environments,
        config.action_bins,
    )
    if checkpoint_data:
        policy = restore_policy(
            checkpoint_data,
            input_dims,
            output_dims,
            output_activations,
            current_task_id,
        )
    else:
        policy = Policy(
            input_dims,
            config.hidden_dims,
            output_dims,
            output_activations,
            shared_output=config.shared_output,
        )

    saved_observation_stats = None
    saved_horizons = None
    if checkpoint_data:
        saved_observation_stats = checkpoint_data["observation_stats"]
        saved_horizons = checkpoint_data["max_steps_by_task"]
        if len(saved_observation_stats) != current_task_id:
            raise ValueError(
                "Checkpoint observation-stat count does not match its task count"
            )
        if len(saved_horizons) != current_task_id:
            raise ValueError(
                "Checkpoint horizon count does not match its task count"
            )

    normalizer = ObservationNormalizer(
        policy.input_dims,
        config.normalize_observations,
        saved_observation_stats,
    )
    horizons = HorizonManager(
        environments,
        adaptive=config.adaptive_max_steps,
        start=config.adaptive_max_steps_start,
        threshold=config.adaptive_max_steps_threshold,
        ratio=config.adaptive_max_steps_ratio,
        saved_limits=saved_horizons,
    )

    run_name = build_run_name(
        env_names,
        current_task_id,
        multi_task,
        config,
    )
    csv_logger = CsvLogger(
        f"logs/csv/{run_name}.csv",
        log_fields(env_names, current_task_id, multi_task, config.fwt),
    )

    noise = NoiseGenerator(config.seed)
    evaluation_rng = np.random.default_rng(config.seed + 1)
    previous_task_ids = list(range(current_task_id)) if not multi_task else []
    total_env_steps = 0
    best_reward = float("-inf")
    best_theta = None
    best_observation_stats = None
    best_horizons = None

    theta = get_flat_params(policy, current_task_id)
    print(
        f"Policy initialized for task {env_names[current_task_id]} with "
        f"input dims {policy.input_dims}, hidden dims {policy.hidden_dims}, "
        f"output dims {policy.output_dims}, theta shape {theta.shape}"
    )
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.alpha,
        weight_decay=config.weight_decay,
    )

    ray.init(address=ray_address)
    print("Ray initialized")
    worker_count = min(
        config.num_workers,
        max(config.num_directions, config.replay_directions),
    )
    workers = [
        ESWorker.remote(
            env_names=env_names,
            noise=noise,
            hidden_sizes=policy.hidden_dims,
            shared_output=policy.shared_output,
            action_bins=config.action_bins,
            normalize_observations=config.normalize_observations,
            obs_clip=config.obs_clip,
        )
        for _ in range(worker_count)
    ]
    print(
        f"Workers initialized: {len(workers)} "
        f"(configured maximum: {config.num_workers})"
    )

    initial_episode_seeds = evaluation_rng.integers(
        0,
        2**32,
        size=10,
    ).tolist()
    initial_log_row = {
        "iteration": 0,
        "max_steps": horizons.for_task(current_task_id),
        "total_steps": 0,
        "task": env_names[current_task_id],
    }
    print("Initial policy evaluation")
    _add_evaluations_to_log_row(
        initial_log_row,
        policy=policy,
        environments=environments,
        env_names=env_names,
        current_task_id=current_task_id,
        multi_task=multi_task,
        config=config,
        episode_seeds=initial_episode_seeds,
        normalizer=normalizer,
    )
    csv_logger.write(initial_log_row)

    for iteration in range(config.iterations):
        training_max_steps = horizons.for_task(current_task_id)
        obs_mean, obs_std = normalizer.snapshot(current_task_id)
        results = collect_rollouts(
            workers,
            noise,
            num_directions=config.num_directions,
            task_id=current_task_id,
            theta=theta,
            sigma=config.sigma,
            max_steps=training_max_steps,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        gradient, flat_rewards, all_steps = compute_gradient(
            results,
            theta.shape[0],
            config.rank_function,
            noise,
            config.sigma,
        )
        for result in results:
            normalizer.increment(
                current_task_id,
                result[3],
                result[4],
                result[5],
            )

        replay_gradient, replay_total_steps = compute_replay_gradient(
            iteration=iteration,
            workers=workers,
            noise=noise,
            policy=policy,
            current_task_id=current_task_id,
            previous_task_ids=previous_task_ids,
            current_gradient=gradient,
            horizons=horizons,
            normalizer=normalizer,
            config=config,
            env_names=env_names,
        )
        gradient += replay_gradient

        apply_es_gradient(
            policy,
            optimizer,
            gradient,
            current_task_id,
            config.frozen_hidden,
        )

        theta = get_flat_params(policy, current_task_id)
        
        rewards_tensor = torch.as_tensor(flat_rewards)
        avg_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        max_reward = max(flat_rewards)

        current_episode_lengths = [
            length for batch in all_steps for length in batch
        ]
        episode_lengths = current_episode_lengths.copy()
        episode_lengths.extend(replay_total_steps)
        total_env_steps += sum(episode_lengths)

        log_row = {
            "iteration": iteration + 1,
            "avg_reward": avg_reward.item(),
            "std_reward": std_reward.item(),
            "max_reward": max_reward,
            "max_steps": training_max_steps,
            "total_steps": total_env_steps,
            "task": env_names[current_task_id],
        }

        print(
            f"Iter {iteration + 1:03d}: avg raw reward = {avg_reward:.2f} "
            f"± {std_reward:.2f}, max = {max_reward:.2f}, "
            f"max_steps = {training_max_steps}, total steps = "
            f"{total_env_steps}, env = {env_names[current_task_id]}"
        )

        if (iteration + 1) % 10 == 0:
            runs = 10
            episode_seeds = evaluation_rng.integers(
                0,
                2**32,
                size=runs,
            ).tolist()

            _add_evaluations_to_log_row(
                log_row,
                policy=policy,
                environments=environments,
                env_names=env_names,
                current_task_id=current_task_id,
                multi_task=multi_task,
                config=config,
                episode_seeds=episode_seeds,
                normalizer=normalizer,
            )

        csv_logger.write(log_row)

        horizon_update = horizons.update(
            current_task_id,
            current_episode_lengths,
        )
        if horizon_update:
            print(
                f"Increased max_steps for {env_names[current_task_id]} from "
                f"{horizon_update.old_limit} to {horizon_update.new_limit}; "
                f"{horizon_update.fraction_hitting_limit:.1%} of rollouts "
                f"hit the cutoff"
            )

        if avg_reward > best_reward and not multi_task:
            best_reward = avg_reward
            best_theta = theta.clone()
            best_observation_stats = normalizer.state_dict()
            best_horizons = horizons.state_dict()

        if multi_task:
            current_task_id = (current_task_id + 1) % len(env_names)
            theta = get_flat_params(policy, current_task_id)

        if (
            config.checkpoint_interval != 0
            and (iteration + 1) % config.checkpoint_interval == 0
        ):
            save_checkpoint(
                f"chkpts/checkpoint_{run_name}_at{iteration + 1}.pth",
                env_names=env_names,
                policy=policy,
                config=config,
                observation_stats=normalizer.state_dict(),
                max_steps_by_task=horizons.state_dict(),
            )
    
    if multi_task:
        best_theta = theta.clone()
        best_observation_stats = normalizer.state_dict()
        best_horizons = horizons.state_dict()
    
    set_flat_params(policy, best_theta, current_task_id)
    print(
        f"Training completed. Best theta saved for task "
        f"{env_names[current_task_id]}."
    )
    save_checkpoint(
        f"chkpts/best_policy_{run_name}.pth",
        env_names=env_names,
        policy=policy,
        config=config,
        observation_stats=best_observation_stats,
        max_steps_by_task=best_horizons,
    )
