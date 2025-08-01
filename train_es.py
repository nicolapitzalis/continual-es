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
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--adaptive-max-steps", type=str2bool, default=True)
    parser.add_argument("--rank-function", type=str, default="centered",
                        choices=["centered", "weighted", "z_score", "fitness_shaping", "none"])
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=str, nargs='*', default=[])
    parser.add_argument("--old-tasks", type=str, nargs='*', default=[])
    parser.add_argument("--all-tasks", type=str, nargs='*', default=[])
    args = parser.parse_args()

    print(f"\n{args}\n")

    # number of samples = num_workers * batch_size
    # Build unified tasks parameter
    if args.all_tasks:
        tasks = args.all_tasks
    elif args.old_tasks and args.resume_from_checkpoint:
        tasks = {
            "env_name": args.env,
            "old_tasks": args.old_tasks,
            "resume_from_checkpoint": args.resume_from_checkpoint
        }
    else:
        tasks = args.env

    train(
        tasks=tasks,
        sigma=args.sigma,
        alpha=args.alpha,
        iterations=args.iterations,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        ray_address=args.ray_address,
        adaptive_max_steps=args.adaptive_max_steps,
        rank_function=args.rank_function,
        checkpoint_interval=args.checkpoint_interval
    )

