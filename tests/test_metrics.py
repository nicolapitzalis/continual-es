import unittest

from es.config import TrainingConfig
from es.metrics import build_run_name, log_fields


class MetricsTest(unittest.TestCase):
    def test_run_name_contains_seed(self):
        config = TrainingConfig(
            seed=17,
            num_directions=250,
            replay_directions=18,
            replay_interval=5,
        )
        run_name = build_run_name(["Task-v0"], 0, False, config)
        self.assertIn("_d250_", run_name)
        self.assertTrue(run_name.endswith("_seed17"))

        continual_name = build_run_name(
            ["Old-v0", "Task-v0"],
            1,
            False,
            config,
        )
        self.assertIn("_replay18_every5", continual_name)

    def test_logs_record_current_task_in_every_training_mode(self):
        single_fields = log_fields(["First-v0"], 0, False, False)
        multitask_fields = log_fields(
            ["First-v0", "Second-v0"],
            0,
            True,
            False,
        )
        self.assertIn("task", single_fields)
        self.assertIn("task", multitask_fields)
        self.assertNotIn("env", multitask_fields)


if __name__ == "__main__":
    unittest.main()
