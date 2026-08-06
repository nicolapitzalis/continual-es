from es.rollout import evaluate_policy


def evaluate_tasks(
    policy,
    environments,
    task_ids,
    *,
    action_bins,
    episode_seeds,
    normalizer,
    obs_clip,
):
    rewards = {}
    for task_id in task_ids:
        obs_mean, obs_std = normalizer.snapshot(task_id)
        rewards[task_id] = evaluate_policy(
            policy,
            environments,
            task_id,
            action_bins,
            episode_seeds,
            obs_mean,
            obs_std,
            obs_clip,
        )
    return rewards

