import tempfile
import unittest
from pathlib import Path

import torch

from es.checkpoint import (
    load_checkpoint,
    restore_policy,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from es.config import TrainingConfig
from es.envs import identity
from es.normalization import ObservationNormalizer
from es.policy import Policy


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_and_task_expansion(self):
        config = TrainingConfig(
            hidden_dims=(4, 4),
            adaptive_max_steps=True,
            normalize_observations=True,
        )
        policy = Policy([3], [4, 4], [2], [identity])
        normalizer = ObservationNormalizer([3], enabled=True)

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                str(path),
                env_names=["OldTask-v0"],
                policy=policy,
                config=config,
                observation_stats=normalizer.state_dict(),
                max_steps_by_task=[200],
            )
            checkpoint = load_checkpoint(str(path))

        validate_checkpoint_compatibility(checkpoint, config)
        restored = restore_policy(
            checkpoint,
            input_dims=[3, 5],
            output_dims=[2, 6],
            output_activations=[identity, identity],
            current_task_id=1,
        )
        self.assertEqual(checkpoint["env_names"], ["OldTask-v0"])
        self.assertEqual(checkpoint["max_steps_by_task"], [200])
        self.assertEqual(restored.input_dims, [3, 5])
        self.assertEqual(restored.output_dims, [2, 6])
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[name], value), name)


if __name__ == "__main__":
    unittest.main()
