import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from es.plotting import (
    load_baseline_maxima,
    load_curve_data,
    normalize_curve_data,
    plot_learning_curves,
    save_figure,
)


class PlottingTest(unittest.TestCase):
    def test_continual_timeline_keeps_evaluations_sparse(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            self._write_log(first, [10, 20], [5.0, None])
            self._write_log(
                second,
                [6, 15],
                [8.0, None],
                old_evaluations={"First-v0_avg_reward": [4.0, None]},
            )

            data = load_curve_data(
                [first, second],
                ["First-v0", "Second-v0"],
                x_axis="steps",
            )

        self.assertEqual(data.training["x"].tolist(), [10, 20, 26, 35])
        self.assertEqual(data.evaluation["x"].tolist(), [10, 26, 26])
        retained = data.evaluation[data.evaluation["role"] == "retained"]
        self.assertEqual(retained["task"].tolist(), ["First-v0"])
        self.assertEqual(retained["reward"].tolist(), [4.0])
        self.assertEqual(data.boundaries, (20.0,))
        self.assertEqual(data.tasks, ("First-v0", "Second-v0"))

    def test_normalization_uses_single_task_evaluation_maxima(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            log = Path(directory) / "log.csv"
            baseline = Path(directory) / "baseline.csv"
            self._write_log(log, [10, 20], [5.0, 10.0])
            self._write_log(baseline, [10, 20], [10.0, 20.0])

            data = load_curve_data([log], ["Task-v0"])
            maxima = load_baseline_maxima({"Task-v0": baseline})
            normalized = normalize_curve_data(data, maxima)

        self.assertEqual(maxima, {"Task-v0": 20.0})
        self.assertEqual(normalized.training["mean"].tolist(), [0.5, 1.0])
        self.assertEqual(normalized.training["std"].tolist(), [0.05, 0.1])
        self.assertEqual(normalized.evaluation["reward"].tolist(), [0.25, 0.5])

    def test_loads_single_file_multitask_log(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            log = Path(directory) / "multitask.csv"
            pd.DataFrame(
                {
                    "iteration": [1, 2, 3, 4],
                    "avg_reward": [1.0, 2.0, 3.0, 4.0],
                    "std_reward": [0.1, 0.2, 0.3, 0.4],
                    "total_steps": [10, 20, 30, 40],
                    "env": ["First-v0", "Second-v0"] * 2,
                    "First-v0_eval_curr_policy": [5.0, None, 7.0, None],
                    "Second-v0_eval_curr_policy": [6.0, None, 8.0, None],
                }
            ).to_csv(log, index=False)

            data = load_curve_data([log], x_axis="iteration")

        self.assertEqual(data.tasks, ("First-v0", "Second-v0"))
        self.assertEqual(
            data.training["task"].tolist(),
            ["First-v0", "Second-v0", "First-v0", "Second-v0"],
        )
        self.assertEqual(len(data.evaluation), 4)
        self.assertEqual(data.phases, ())

    def test_plot_can_be_saved(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            log = Path(directory) / "log.csv"
            output = Path(directory) / "curves.png"
            self._write_log(log, [10, 20], [5.0, 10.0])
            data = load_curve_data([log], ["Task-v0"])
            figure = plot_learning_curves(data, smooth=2)
            save_figure(figure, output)
            self.assertGreater(output.stat().st_size, 0)

    @staticmethod
    def _write_log(path, total_steps, evaluations, old_evaluations=None):
        columns = {
            "iteration": [1, 2],
            "avg_reward": [10.0, 20.0],
            "std_reward": [1.0, 2.0],
            "max_reward": [12.0, 24.0],
            "total_steps": total_steps,
            "eval_curr_policy": evaluations,
        }
        columns.update(old_evaluations or {})
        pd.DataFrame(columns).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
