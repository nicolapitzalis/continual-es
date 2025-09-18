import argparse
from core.es_master import train

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, nargs='*', default=[])
    parser.add_argument("--to-train", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[64, 64])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--rank-function", type=str, default="centered",
                        choices=["centered", "weighted", "z_score", "fitness_shaping", "none"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--adaptive-max-steps", type=str2bool, default=True)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--shared-output", type=str2bool, default=False,
                        help="If True, all tasks share the same output layer. If False, each task has its own output layer.")
    parser.add_argument("--replay-batch-size", type=int, default=0)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--frozen-hidden", type=str2bool, default=False,
                        help="If True, the hidden layers are frozen during training. If False, they are updated.")
    args = parser.parse_args()

    print(f"\n{args}\n")

    train(
        envs=args.envs,
        to_train=args.to_train,
        sigma=args.sigma,
        alpha=args.alpha,
        hidden_dims=args.hidden_dims,
        iterations=args.iterations,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        rank_function=args.rank_function,
        ray_address=args.ray_address,
        adaptive_max_steps=args.adaptive_max_steps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint=args.checkpoint,
        shared_output=args.shared_output,
        replay_batch_size=args.replay_batch_size,
        replay_weight=args.replay_weight,
        frozen_hidden=args.frozen_hidden
    )

