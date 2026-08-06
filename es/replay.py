import torch

from es.policy import get_flat_params
from es.sampling import collect_rollouts
from es.updates import compute_gradient, repack_replay_gradient


def replay_due(iteration, interval):
    """Return whether a zero-based generation should include replay."""
    return (iteration + 1) % interval == 0


def compute_replay_gradient(
    *,
    iteration,
    workers,
    noise,
    policy,
    current_task_id,
    previous_task_ids,
    current_gradient,
    horizons,
    normalizer,
    config,
    env_names,
):
    if (
        config.replay_directions == 0
        or not previous_task_ids
        or not replay_due(iteration, config.replay_interval)
    ):
        return torch.zeros_like(current_gradient), []

    combined_gradient = torch.zeros_like(current_gradient)
    total_steps = []
    for replay_task_id in previous_task_ids:
        replay_theta = get_flat_params(policy, replay_task_id)
        obs_mean, obs_std = normalizer.snapshot(replay_task_id)
        results = collect_rollouts(
            workers,
            noise,
            num_directions=config.replay_directions,
            task_id=replay_task_id,
            theta=replay_theta,
            sigma=config.sigma,
            max_steps=horizons.for_task(replay_task_id),
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        task_gradient, rewards, steps = compute_gradient(
            results,
            replay_theta.shape[0],
            config.rank_function,
            noise,
            config.sigma,
        )
        for result in results:
            normalizer.increment(
                replay_task_id,
                result[3],
                result[4],
                result[5],
            )
        combined_gradient += repack_replay_gradient(
            task_gradient,
            policy,
            replay_task_id,
            current_task_id,
        )
        total_steps.extend(length for batch in steps for length in batch)
        mean_reward = torch.as_tensor(rewards).mean().item()
        print(
            f"Replay for env {env_names[replay_task_id]}: "
            f"avg reward = {mean_reward:.2f}"
        )

    combined_gradient *= config.replay_weight / len(previous_task_ids)
    return combined_gradient, total_steps
