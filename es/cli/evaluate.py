import argparse

import numpy as np

from es.checkpoint import load_checkpoint
from es.envs import make_env
from es.normalization import ObservationNormalizer
from es.policy import Policy
from es.rollout import evaluate_policy


def resolve_task_id(task, env_names):
    if task is None:
        return len(env_names) - 1
    try:
        task_id = int(task)
    except ValueError:
        if task not in env_names:
            raise ValueError(
                f"Unknown task {task!r}; checkpoint tasks are {env_names}"
            )
        return env_names.index(task)
    if not 0 <= task_id < len(env_names):
        raise ValueError(
            f"Task index {task_id} is outside [0, {len(env_names) - 1}]"
        )
    return task_id


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved evolution-strategies policy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task",
        help="Environment name or task index; defaults to the last task.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    checkpoint = load_checkpoint(args.checkpoint)
    env_names = list(checkpoint["env_names"])
    task_id = resolve_task_id(args.task, env_names)
    environments = [
        make_env(
            env_name,
            render_mode="human" if args.render and index == task_id else None,
        )
        for index, env_name in enumerate(env_names)
    ]
    try:
        policy = Policy(
            checkpoint["input_dims"],
            checkpoint["hidden_dims"],
            checkpoint["output_dims"],
            checkpoint["output_activation"],
            checkpoint["shared_output"],
        )
        policy.load_state_dict(checkpoint["state_dict"])
        normalizer = ObservationNormalizer(
            policy.input_dims,
            checkpoint["normalize_observations"],
            checkpoint["observation_stats"],
        )
        obs_mean, obs_std = normalizer.snapshot(task_id)
        rng = np.random.default_rng(args.seed)
        episode_seeds = rng.integers(0, 2**32, size=args.episodes).tolist()
        reward = evaluate_policy(
            policy,
            environments,
            task_id,
            checkpoint["action_bins"],
            episode_seeds,
            obs_mean,
            obs_std,
            checkpoint["obs_clip"],
        )
    finally:
        for environment in environments:
            environment.close()

    print(
        f"Average reward over {args.episodes} episodes on "
        f"{env_names[task_id]}: {reward:.2f}"
    )


if __name__ == "__main__":
    main()

