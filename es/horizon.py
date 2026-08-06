from dataclasses import dataclass

import numpy as np


def native_max_steps(env):
    spec = getattr(env, "spec", None)
    limit = getattr(spec, "max_episode_steps", None)
    return None if limit is None else int(limit)


@dataclass(frozen=True)
class HorizonUpdate:
    old_limit: int | None
    new_limit: int | None
    fraction_hitting_limit: float


class HorizonManager:
    def __init__(
        self,
        envs,
        *,
        adaptive,
        start,
        threshold,
        ratio,
        saved_limits=None,
    ):
        self.adaptive = adaptive
        self.start = int(start)
        self.threshold = float(threshold)
        self.ratio = float(ratio)
        self.native_limits = [native_max_steps(env) for env in envs]

        previous_count = 0 if saved_limits is None else len(saved_limits)
        if previous_count > len(envs):
            raise ValueError("More saved horizons than environments")

        if not adaptive:
            self.limits = [None] * len(envs)
            return

        self.limits = []
        for task_id, native_limit in enumerate(self.native_limits):
            if task_id < previous_count:
                saved_limit = saved_limits[task_id]
                if saved_limit is not None and saved_limit <= 0:
                    raise ValueError("Checkpoint contains an invalid max-step value")
                self.limits.append(self._effective_limit(saved_limit, native_limit))
            else:
                self.limits.append(self._effective_limit(self.start, native_limit))

    @staticmethod
    def _effective_limit(limit, native_limit):
        if limit is None:
            return None
        if native_limit is not None and limit >= native_limit:
            return None
        return int(limit)

    def for_task(self, task_id):
        return self.limits[task_id]

    def update(self, task_id, episode_lengths):
        current_limit = self.limits[task_id]
        if not self.adaptive or current_limit is None:
            return None

        lengths = np.asarray(episode_lengths, dtype=np.int64)
        if lengths.size == 0:
            raise ValueError("Cannot update a horizon without episode lengths")

        fraction = float(np.mean(lengths == current_limit))
        if fraction < self.threshold:
            return None

        increased = max(current_limit + 1, int(current_limit * self.ratio))
        new_limit = self._effective_limit(
            increased,
            self.native_limits[task_id],
        )
        self.limits[task_id] = new_limit
        return HorizonUpdate(current_limit, new_limit, fraction)

    def state_dict(self):
        return self.limits.copy()

