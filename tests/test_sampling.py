import unittest
from unittest.mock import patch

from es.sampling import distribute_directions
from es.sampling import collect_rollouts
from es.replay import replay_due


class _RemoteEvaluate:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return args


class _Worker:
    def __init__(self):
        self.evaluate = _RemoteEvaluate()


class _Noise:
    def __init__(self):
        self.counts = []

    def sample_indices_and_episode_seeds(self, count):
        self.counts.append(count)
        return list(range(count)), list(range(100, 100 + count))


class DirectionDistributionTest(unittest.TestCase):
    def test_uses_fewer_workers_when_directions_are_fewer(self):
        self.assertEqual(distribute_directions(3, 8), [1, 1, 1])

    def test_distributes_remainder_without_losing_directions(self):
        counts = distribute_directions(250, 112)
        self.assertEqual(len(counts), 112)
        self.assertEqual(sum(counts), 250)
        self.assertEqual(counts.count(3), 26)
        self.assertEqual(counts.count(2), 86)

    def test_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            distribute_directions(0, 4)
        with self.assertRaises(ValueError):
            distribute_directions(4, 0)

    def test_replay_interval_uses_one_based_generation_numbers(self):
        replay_generations = [
            iteration + 1
            for iteration in range(12)
            if replay_due(iteration, 5)
        ]
        self.assertEqual(replay_generations, [5, 10])

    def test_collect_rollouts_calls_only_active_workers(self):
        workers = [_Worker() for _ in range(8)]
        noise = _Noise()
        with patch("es.sampling.ray.get", side_effect=lambda futures: futures):
            results = collect_rollouts(
                workers,
                noise,
                num_directions=3,
                task_id=0,
                theta=None,
                sigma=0.1,
                max_steps=None,
                obs_mean=None,
                obs_std=None,
            )

        self.assertEqual(noise.counts, [1, 1, 1])
        self.assertEqual(len(results), 3)
        self.assertEqual(
            [len(worker.evaluate.calls) for worker in workers],
            [1, 1, 1, 0, 0, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
