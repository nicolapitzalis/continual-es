import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from es.plotting import (
    load_baseline_maxima,
    load_curve_data,
    normalize_curve_data,
    plot_learning_curves,
    save_figure,
)


def baseline_assignment(value):
    try:
        task, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "baseline must have the form TASK=CSV"
        ) from error
    if not task or not path:
        raise argparse.ArgumentTypeError("baseline must have the form TASK=CSV")
    return task, path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot evolution-strategies training results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves = subparsers.add_parser(
        "curves",
        help="Plot perturbed-policy and clean-evaluation learning curves.",
    )
    curves.add_argument(
        "logs",
        nargs="+",
        help="Ordered CSV logs, one per continual task or one multitask log.",
    )
    curves.add_argument(
        "--tasks",
        nargs="+",
        help="Task names in log order; inferred when possible.",
    )
    curves.add_argument(
        "--series",
        choices=("perturbed", "evaluation", "both"),
        default="both",
    )
    curves.add_argument(
        "--scale",
        choices=("raw", "normalized", "both"),
        default="raw",
    )
    curves.add_argument(
        "--baseline",
        action="append",
        type=baseline_assignment,
        default=[],
        metavar="TASK=CSV",
        help=(
            "Single-task log defining a task's normalization maximum; "
            "repeat once per task."
        ),
    )
    curves.add_argument(
        "--x-axis",
        choices=("iteration", "steps"),
        default="steps",
    )
    curves.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Centered rolling window for perturbed returns (default: 1).",
    )
    curves.add_argument(
        "--band",
        choices=("std", "none"),
        default="std",
        help="Uncertainty band for perturbed-policy returns.",
    )
    curves.add_argument("--max-iterations", type=int)
    curves.add_argument("--title")
    curves.add_argument(
        "--output",
        default="plots/learning_curves",
        help="Output path or path stem (default: plots/learning_curves).",
    )
    curves.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        default="pdf",
    )
    curves.add_argument("--dpi", type=int, default=300)
    curves.add_argument("--show", action="store_true")
    curves.set_defaults(run=run_curves)
    return parser


def run_curves(args):
    if args.smooth < 1:
        raise ValueError("--smooth must be positive")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    data = load_curve_data(
        args.logs,
        args.tasks,
        x_axis=args.x_axis,
        max_iterations=args.max_iterations,
    )
    scales = ("raw", "normalized") if args.scale == "both" else (args.scale,)

    normalized_data = None
    if "normalized" in scales:
        baseline_paths = dict(args.baseline)
        maxima = load_baseline_maxima(baseline_paths)
        normalized_data = normalize_curve_data(data, maxima)

    saved_paths = []
    for scale in scales:
        plot_data = normalized_data if scale == "normalized" else data
        figure = plot_learning_curves(
            plot_data,
            series=args.series,
            smooth=args.smooth,
            band=args.band,
            title=args.title,
        )
        output_path = _resolve_output_path(
            args.output,
            args.format,
            scale,
            multiple=len(scales) > 1,
        )
        saved_paths.append(save_figure(figure, output_path, dpi=args.dpi))
        if not args.show:
            plt.close(figure)

    if args.show:
        plt.show()
    for path in saved_paths:
        print(f"Saved {path}")


def _resolve_output_path(output, output_format, scale, *, multiple):
    path = Path(output)
    if path.suffix:
        output_format = path.suffix.lstrip(".")
        path = path.with_suffix("")
    suffix = f"_{scale}" if multiple else ""
    return path.parent / f"{path.name}{suffix}.{output_format}"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

