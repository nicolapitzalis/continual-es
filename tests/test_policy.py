import unittest

import torch

from es.envs import identity
from es.policy import Policy, add_task, get_flat_params, set_flat_params


class PolicyTest(unittest.TestCase):
    def test_add_distinct_task_preserves_old_parameters(self):
        policy = Policy([3], [4, 4], [2], [identity])
        old_state = {
            name: value.clone() for name, value in policy.state_dict().items()
        }

        add_task(policy, 5, 6, identity)

        for name, value in old_state.items():
            self.assertTrue(torch.equal(policy.state_dict()[name], value), name)
        self.assertEqual(policy(torch.zeros(1, 3), 0).shape, (1, 2))
        self.assertEqual(policy(torch.zeros(1, 5), 1).shape, (1, 6))

    def test_shared_output_expansion_preserves_existing_rows(self):
        policy = Policy([3], [4, 4], [2], [identity], shared_output=True)
        old_weight = policy.output[0].weight.clone()
        old_bias = policy.output[0].bias.clone()

        add_task(policy, 5, 6, identity)

        self.assertTrue(torch.equal(policy.output[0].weight[:2], old_weight))
        self.assertTrue(torch.equal(policy.output[0].bias[:2], old_bias))
        self.assertEqual(policy(torch.zeros(1, 3), 0).shape, (1, 2))
        self.assertEqual(policy(torch.zeros(1, 5), 1).shape, (1, 6))

    def test_flat_parameter_round_trip(self):
        policy = Policy([3], [4, 4], [2], [identity])
        flat = get_flat_params(policy, 0)
        replacement = torch.arange(flat.numel(), dtype=flat.dtype)
        set_flat_params(policy, replacement, 0)
        self.assertTrue(torch.equal(get_flat_params(policy, 0), replacement))


if __name__ == "__main__":
    unittest.main()

