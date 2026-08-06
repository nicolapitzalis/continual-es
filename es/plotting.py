from dataclasses import dataclass, replace
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CURVE_COLUMNS = {
    "iteration",
    "avg_reward",
    "std_reward",
    "total_steps",
}
TASK_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)


@dataclass(frozen=True)
class TaskPhase:
    task: str
    start: float
    end: float


@dataclass(frozen=True)
class CurveData:
    training: pd.DataFrame
    evaluation: pd.DataFrame
    tasks: tuple[str, ...]
    phases: tuple[TaskPhase, ...]
    boundaries: tuple[float, ...]
    x_axis: str
    normalized: bool = False


def load_curve_data(
    log_paths,
    tasks=None,
    *,
    x_axis="steps",
    max_iterations=None,
):
    """Load single-task, continual, or multitask logs for curve plotting."""
    if x_axis not in {"iteration", "steps"}:
        raise ValueError("x_axis must be 'iteration' or 'steps'")
    if not log_paths:
        raise ValueError("At least one CSV log is required")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    paths = [Path(path) for path in log_paths]
    frames = [pd.read_csv(path) for path in paths]
    for path, frame in zip(paths, frames):
        missing = CURVE_COLUMNS.difference(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {names}")

    task_column = _multitask_column(frames[0]) if len(frames) == 1 else None
    is_multitask = task_column is not None
    if is_multitask:
        return _load_multitask_curve_data(
            frames[0],
            tasks,
            task_column=task_column,
            x_axis=x_axis,
            max_iterations=max_iterations,
        )

    if tasks is None:
        tasks = [
            _infer_task(path, frame)
            for path, frame in zip(paths, frames)
        ]
    if len(tasks) != len(paths):
        raise ValueError(
            "Sequential curves require one task name per CSV log; "
            f"received {len(tasks)} tasks and {len(paths)} logs"
        )

    x_column = "iteration" if x_axis == "iteration" else "total_steps"
    training_parts = []
    evaluation_parts = []
    phases = []
    boundaries = []
    offset = 0.0

    for segment, (task, path, frame) in enumerate(zip(tasks, paths, frames)):
        frame = _limit_iterations(frame, max_iterations)
        if frame.empty:
            raise ValueError(f"{path} has no rows after filtering")

        local_x = _numeric_column(frame, x_column, path)
        if not local_x.is_monotonic_increasing:
            raise ValueError(f"{x_column} must be increasing in {path}")
        x = local_x + offset

        training = pd.DataFrame(
            {
                "x": x,
                "task": task,
                "segment": segment,
                "mean": pd.to_numeric(frame["avg_reward"], errors="coerce"),
                "std": pd.to_numeric(frame["std_reward"], errors="coerce"),
            }
        ).dropna(subset=["x", "mean"])
        training_parts.append(training)

        if "eval_curr_policy" in frame.columns:
            evaluation = pd.DataFrame(
                {
                    "x": x,
                    "task": task,
                    "segment": segment,
                    "role": "current",
                    "reward": pd.to_numeric(
                        frame["eval_curr_policy"],
                        errors="coerce",
                    ),
                }
            ).dropna(subset=["x", "reward"])
            evaluation_parts.append(evaluation)

        for previous_task in tasks[:segment]:
            column = f"{previous_task}_avg_reward"
            if column not in frame.columns:
                continue
            retained_evaluation = pd.DataFrame(
                {
                    "x": x,
                    "task": previous_task,
                    "segment": segment,
                    "role": "retained",
                    "reward": pd.to_numeric(frame[column], errors="coerce"),
                }
            ).dropna(subset=["x", "reward"])
            evaluation_parts.append(retained_evaluation)

        phase_end = float(x.iloc[-1])
        phases.append(TaskPhase(str(task), offset, phase_end))
        if segment < len(frames) - 1:
            boundaries.append(phase_end)
        offset = phase_end

    return CurveData(
        training=pd.concat(training_parts, ignore_index=True),
        evaluation=_concat_or_empty(
            evaluation_parts,
            columns=("x", "task", "segment", "role", "reward"),
        ),
        tasks=tuple(str(task) for task in tasks),
        phases=tuple(phases),
        boundaries=tuple(boundaries),
        x_axis=x_axis,
    )


def load_baseline_maxima(task_to_path):
    """Return each task's best clean evaluation from a single-task log."""
    maxima = {}
    for task, path_value in task_to_path.items():
        path = Path(path_value)
        frame = pd.read_csv(path)
        if "eval_curr_policy" in frame.columns:
            column = "eval_curr_policy"
        elif f"{task}_eval_curr_policy" in frame.columns:
            column = f"{task}_eval_curr_policy"
        else:
            raise ValueError(
                f"{path} has no evaluation column for task {task!r}"
            )

        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"{path} has no evaluation values for task {task!r}")
        maximum = float(values.max())
        if np.isclose(maximum, 0.0):
            raise ValueError(
                f"Cannot normalize task {task!r}: its baseline maximum is zero"
            )
        maxima[str(task)] = maximum
    return maxima


def normalize_curve_data(data, baseline_maxima):
    """Normalize every task so its single-task maximum has value one."""
    missing = set(data.tasks).difference(baseline_maxima)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing normalization baselines for: {names}")

    training = data.training.copy()
    references = training["task"].map(baseline_maxima)
    training["mean"] = training["mean"] / references
    training["std"] = training["std"] / references.abs()

    evaluation = data.evaluation.copy()
    if not evaluation.empty:
        evaluation["reward"] = (
            evaluation["reward"]
            / evaluation["task"].map(baseline_maxima)
        )

    return replace(
        data,
        training=training,
        evaluation=evaluation,
        normalized=True,
    )


def plot_learning_curves(
    data,
    *,
    series="both",
    smooth=1,
    band="std",
    title=None,
):
    """Plot perturbed-policy returns and sparse clean evaluations."""
    if series not in {"perturbed", "evaluation", "both"}:
        raise ValueError("series must be 'perturbed', 'evaluation', or 'both'")
    if band not in {"std", "none"}:
        raise ValueError("band must be 'std' or 'none'")
    if smooth < 1:
        raise ValueError("smooth must be positive")
    if series in {"evaluation", "both"} and data.evaluation.empty:
        raise ValueError("The supplied logs contain no clean evaluations")

    panel_names = (
        ["perturbed", "evaluation"] if series == "both" else [series]
    )
    height = 7.6 if len(panel_names) == 2 else 4.8
    fig, axes = plt.subplots(
        len(panel_names),
        1,
        figsize=(11, height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    colors = {
        task: TASK_COLORS[index % len(TASK_COLORS)]
        for index, task in enumerate(data.tasks)
    }

    for axis, panel in zip(axes, panel_names):
        _decorate_axis(axis, data, colors)
        if panel == "perturbed":
            _plot_perturbed(axis, data, colors, smooth, band)
            axis.set_title("Perturbed policies · training horizon", loc="left")
        else:
            _plot_evaluation(axis, data, colors)
            evaluation_title = "Unperturbed policy evaluation · full horizon"
            axis.set_title(evaluation_title, loc="left")

        ylabel = "Normalized average return" if data.normalized else "Average return"
        axis.set_ylabel(ylabel)
        if data.normalized:
            axis.axhline(1.0, color="#6B7280", linewidth=1, linestyle=":")

    x_label = "Training iteration" if data.x_axis == "iteration" else "Environment steps"
    axes[-1].set_xlabel(x_label)

    handles = [
        Line2D([0], [0], color=colors[task], linewidth=2.4, label=task)
        for task in data.tasks
    ]
    title_text = title or "Evolution strategies learning curves"
    fig.suptitle(title_text, fontsize=16, fontweight="semibold", y=0.995)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(handles), 4),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.0)
    return fig


def save_figure(figure, path, *, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _load_multitask_curve_data(
    frame,
    tasks,
    *,
    task_column,
    x_axis,
    max_iterations,
):
    frame = _limit_iterations(frame, max_iterations)
    if frame.empty:
        raise ValueError("The multitask log has no rows after filtering")

    observed_tasks = tuple(
        dict.fromkeys(frame[task_column].dropna().astype(str))
    )
    tasks = tuple(str(task) for task in (tasks or observed_tasks))
    if not tasks:
        raise ValueError("Could not infer task names from the multitask log")
    unknown = set(frame[task_column].dropna().astype(str)).difference(tasks)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"The multitask log contains unspecified tasks: {names}")

    x_column = "iteration" if x_axis == "iteration" else "total_steps"
    x = _numeric_column(frame, x_column, "multitask log")
    training = pd.DataFrame(
        {
            "x": x,
            "task": frame[task_column].astype(str),
            "segment": 0,
            "mean": pd.to_numeric(frame["avg_reward"], errors="coerce"),
            "std": pd.to_numeric(frame["std_reward"], errors="coerce"),
        }
    ).dropna(subset=["x", "mean"])

    evaluation_parts = []
    for task in tasks:
        column = f"{task}_eval_curr_policy"
        if column not in frame.columns:
            continue
        evaluation_parts.append(
            pd.DataFrame(
                {
                    "x": x,
                    "task": task,
                    "segment": 0,
                    "role": "current",
                    "reward": pd.to_numeric(frame[column], errors="coerce"),
                }
            ).dropna(subset=["x", "reward"])
        )

    return CurveData(
        training=training,
        evaluation=_concat_or_empty(
            evaluation_parts,
            columns=("x", "task", "segment", "role", "reward"),
        ),
        tasks=tasks,
        phases=(),
        boundaries=(),
        x_axis=x_axis,
    )


def _limit_iterations(frame, max_iterations):
    if max_iterations is None:
        return frame.reset_index(drop=True).copy()
    iterations = pd.to_numeric(frame["iteration"], errors="coerce")
    return frame.loc[iterations <= max_iterations].reset_index(drop=True).copy()


def _numeric_column(frame, column, path):
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} contains missing or non-numeric values in {path}")
    return values.reset_index(drop=True)


def _multitask_column(frame):
    for column in ("task", "env"):
        if column not in frame.columns:
            continue
        task_count = frame[column].dropna().astype(str).nunique()
        if task_count > 1:
            return column
    return None


def _infer_task(path, frame):
    if "task" in frame.columns:
        task_names = frame["task"].dropna().astype(str).unique()
        if len(task_names) == 1:
            return task_names[0]

    match = re.match(r"log_(.+?)_s[-+0-9.eE]+_a", path.stem)
    if not match:
        raise ValueError(
            f"Could not infer a task from {path}; provide --tasks explicitly"
        )
    return match.group(1)


def _concat_or_empty(frames, columns):
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns)


def _smooth(values, window):
    return values.rolling(window, center=True, min_periods=1).mean()


def _plot_perturbed(axis, data, colors, smooth, band):
    for (_, task), frame in data.training.groupby(
        ["segment", "task"],
        sort=False,
    ):
        frame = frame.sort_values("x")
        mean = _smooth(frame["mean"], smooth)
        color = colors[task]
        axis.plot(frame["x"], mean, color=color, linewidth=2.1)
        if band == "std":
            std = _smooth(frame["std"].fillna(0.0), smooth)
            axis.fill_between(
                frame["x"],
                mean - std,
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )


def _plot_evaluation(axis, data, colors):
    for (task, role), frame in data.evaluation.groupby(
        ["task", "role"],
        sort=False,
    ):
        frame = frame.sort_values("x")
        axis.plot(
            frame["x"],
            frame["reward"],
            color=colors[task],
            linewidth=2.1,
            linestyle="--" if role == "retained" else "-",
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )


def _decorate_axis(axis, data, colors):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")

    for phase in data.phases:
        axis.axvspan(
            phase.start,
            phase.end,
            color=colors[phase.task],
            alpha=0.035,
            linewidth=0,
        )
    for boundary in data.boundaries:
        axis.axvline(
            boundary,
            color="#6B7280",
            linewidth=1,
            linestyle=(0, (4, 4)),
            alpha=0.8,
        )
