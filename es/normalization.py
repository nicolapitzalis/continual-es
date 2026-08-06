import numpy as np


class RunningObservationStats:
    def __init__(self, dimension, epsilon=1e-2, variance_floor=1e-2):
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.sumsq = np.full(dimension, epsilon, dtype=np.float64)
        self.count = float(epsilon)
        self.variance_floor = float(variance_floor)

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        variance = self.sumsq / self.count - np.square(self.mean)
        return np.sqrt(np.maximum(variance, self.variance_floor))

    def increment(self, obs_sum, obs_sumsq, obs_count):
        if obs_count == 0:
            return
        self.sum += np.asarray(obs_sum, dtype=np.float64)
        self.sumsq += np.asarray(obs_sumsq, dtype=np.float64)
        self.count += int(obs_count)

    def state_dict(self):
        return {
            "sum": self.sum.copy(),
            "sumsq": self.sumsq.copy(),
            "count": self.count,
            "variance_floor": self.variance_floor,
        }

    def load_state_dict(self, state):
        obs_sum = np.asarray(state["sum"], dtype=np.float64)
        obs_sumsq = np.asarray(state["sumsq"], dtype=np.float64)
        if obs_sum.shape != self.sum.shape or obs_sumsq.shape != self.sumsq.shape:
            raise ValueError(
                f"Observation-stat shape mismatch: expected {self.sum.shape}, "
                f"got {obs_sum.shape} and {obs_sumsq.shape}"
            )
        self.sum[:] = obs_sum
        self.sumsq[:] = obs_sumsq
        self.count = float(state["count"])
        self.variance_floor = float(
            state.get("variance_floor", self.variance_floor)
        )


class ObservationNormalizer:
    def __init__(self, dimensions, enabled, saved_states=None):
        self.enabled = enabled
        self.stats = [RunningObservationStats(dim) for dim in dimensions]

        if not enabled or saved_states is None:
            return
        if len(saved_states) > len(self.stats):
            raise ValueError("More saved observation-stat entries than policy tasks")
        for stats, saved_state in zip(self.stats, saved_states):
            stats.load_state_dict(saved_state)

    def snapshot(self, task_id):
        if not self.enabled:
            return None, None
        stats = self.stats[task_id]
        return stats.mean.copy(), stats.std.copy()

    def increment(self, task_id, obs_sum, obs_sumsq, obs_count):
        if self.enabled:
            self.stats[task_id].increment(obs_sum, obs_sumsq, obs_count)

    def state_dict(self):
        return [stats.state_dict() for stats in self.stats]

