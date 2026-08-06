import argparse

from es.config import RANK_FUNCTIONS, TrainingConfig
from es.trainer import train


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a policy with distributed evolution strategies."
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One task for single/continual training, or multiple tasks for "
            "multitask training. With --checkpoint, specify exactly one new task."
        ),
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help=(
            "Maximum concurrent Ray rollout actors; fewer are created when "
            "both direction counts are smaller."
        ),
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=384,
        help=(
            "Total antithetic perturbation directions per generation, "
            "distributed across the available workers."
        ),
    )
    parser.add_argument(
        "--action-bins",
        type=int,
        default=1,
        help="Bins per continuous action component; 1 keeps continuous actions.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument(
        "--rank-function",
        type=str,
        default="centered",
        choices=RANK_FUNCTIONS,
    )
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument(
        "--adaptive-max-steps",
        type=str2bool,
        default=False,
        help=(
            "Use monotonic per-task training cutoffs; clean evaluation always "
            "uses the native environment horizon."
        ),
    )
    parser.add_argument(
        "--adaptive-max-steps-start",
        type=int,
        default=100,
        help="Initial training cutoff for each new task.",
    )
    parser.add_argument(
        "--adaptive-max-steps-threshold",
        type=float,
        default=0.7,
        help="Fraction of rollouts that must hit the cutoff before it increases.",
    )
    parser.add_argument(
        "--adaptive-max-steps-ratio",
        type=float,
        default=2.0,
        help="Multiplicative cutoff increase.",
    )
    parser.add_argument("--normalize-observations", type=str2bool, default=False)
    parser.add_argument("--obs-clip", type=float, default=5.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--shared-output",
        type=str2bool,
        default=False,
        help=(
            "If true, all tasks share one output layer; otherwise each task "
            "has its own output layer."
        ),
    )
    parser.add_argument(
        "--replay-directions",
        type=int,
        default=0,
        help="Total perturbation directions per previous task when replay runs.",
    )
    parser.add_argument(
        "--replay-interval",
        type=int,
        default=1,
        help="Run replay every N generations (default: every generation).",
    )
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument(
        "--frozen-hidden",
        type=str2bool,
        default=False,
        help="If true, shared hidden layers are not updated on the current task.",
    )
    parser.add_argument("--fwt", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"\n{args}\n")

    config = TrainingConfig(
        sigma=args.sigma,
        alpha=args.alpha,
        hidden_dims=tuple(args.hidden_dims),
        iterations=args.iterations,
        num_workers=args.num_workers,
        num_directions=args.num_directions,
        action_bins=args.action_bins,
        weight_decay=args.weight_decay,
        rank_function=args.rank_function,
        adaptive_max_steps=args.adaptive_max_steps,
        adaptive_max_steps_start=args.adaptive_max_steps_start,
        adaptive_max_steps_threshold=args.adaptive_max_steps_threshold,
        adaptive_max_steps_ratio=args.adaptive_max_steps_ratio,
        normalize_observations=args.normalize_observations,
        obs_clip=args.obs_clip,
        checkpoint_interval=args.checkpoint_interval,
        shared_output=args.shared_output,
        replay_directions=args.replay_directions,
        replay_interval=args.replay_interval,
        replay_weight=args.replay_weight,
        frozen_hidden=args.frozen_hidden,
        fwt=args.fwt,
        seed=args.seed,
    )
    train(
        envs=args.env,
        config=config,
        ray_address=args.ray_address,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()

