import unittest

from es.horizon import HorizonManager


class _Spec:
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps


class _Env:
    def __init__(self, max_episode_steps):
        self.spec = _Spec(max_episode_steps)


class HorizonManagerTest(unittest.TestCase):
    def test_cutoff_grows_only_when_threshold_is_met(self):
        manager = HorizonManager(
            [_Env(1000)],
            adaptive=True,
            start=100,
            threshold=0.7,
            ratio=2.0,
        )

        self.assertIsNone(manager.update(0, [100, 20, 30, 40]))
        self.assertEqual(manager.for_task(0), 100)

        update = manager.update(0, [100, 100, 100, 20])
        self.assertEqual(update.old_limit, 100)
        self.assertEqual(update.new_limit, 200)
        self.assertEqual(update.fraction_hitting_limit, 0.75)

    def test_native_horizon_is_represented_by_none(self):
        manager = HorizonManager(
            [_Env(150)],
            adaptive=True,
            start=100,
            threshold=0.5,
            ratio=2.0,
        )
        update = manager.update(0, [100, 100])
        self.assertIsNone(update.new_limit)
        self.assertIsNone(manager.for_task(0))

    def test_saved_tasks_keep_their_limits_and_new_task_starts_fresh(self):
        manager = HorizonManager(
            [_Env(1000), _Env(1000), _Env(1000)],
            adaptive=True,
            start=100,
            threshold=0.7,
            ratio=2.0,
            saved_limits=[400, None],
        )
        self.assertEqual(manager.state_dict(), [400, None, 100])


if __name__ == "__main__":
    unittest.main()

