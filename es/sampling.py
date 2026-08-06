import ray


def distribute_directions(num_directions, num_workers):
    """Split a total direction count over the minimum workers required."""
    if num_directions <= 0:
        raise ValueError("num_directions must be positive")
    if num_workers <= 0:
        raise ValueError("At least one worker is required")

    active_worker_count = min(num_directions, num_workers)
    base, remainder = divmod(num_directions, active_worker_count)
    return [
        base + (worker_id < remainder)
        for worker_id in range(active_worker_count)
    ]


def collect_rollouts(
    workers,
    noise,
    *,
    num_directions,
    task_id,
    theta,
    sigma,
    max_steps,
    obs_mean,
    obs_std,
):
    direction_counts = distribute_directions(num_directions, len(workers))
    active_workers = workers[: len(direction_counts)]
    batches = [
        noise.sample_indices_and_episode_seeds(count)
        for count in direction_counts
    ]
    futures = [
        worker.evaluate.remote(
            task_id,
            theta,
            sigma,
            indices,
            max_steps,
            episode_seeds,
            obs_mean,
            obs_std,
        )
        for worker, (indices, episode_seeds) in zip(active_workers, batches)
    ]
    return ray.get(futures)
