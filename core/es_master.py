import ray
import torch
import csv
import os
from distributed.worker import ESWorker
from core.noise_generator import NoiseGenerator
from core.policy import get_flat_params, set_flat_params, build_policy, extract_and_transfer_hidden
from utils.utils import compute_centered_ranks, compute_weighted_ranks, fitness_shaping, z_score_ranks
from core.updates import apply_weight_decay
from environments.env_utils import make_env
from core.rollout import rollout

def train(
    tasks=[], 
    sigma=0.1, 
    alpha=0.05, 
    iterations=1000, 
    num_workers=12, 
    batch_size=32, 
    weight_decay=0.005, 
    rank_function='centered', 
    ray_address=None, 
    adaptive_max_steps=True, 
    checkpoint_interval=100):
    
    
    ray.init(address=ray_address)
    print("Ray initialized")
    noise = NoiseGenerator(seed=42)

    # Determine mode
    if isinstance(tasks, dict):  # Continual
        env_name = tasks.get("env_name")
        old_tasks = tasks.get("old_tasks", [])
        resume_from_checkpoint = tasks.get("resume_from_checkpoint", [])
        env = make_env(env_name)
        policy = build_policy(env)

        if resume_from_checkpoint and old_tasks:
            theta = torch.load(resume_from_checkpoint[-1])
            old_env = make_env(old_tasks[-1])
            extract_and_transfer_hidden(theta, old_env, policy)
            old_policies = []
            for task, chkpt in zip(old_tasks, resume_from_checkpoint):
                theta = torch.load(chkpt)
                old_env = make_env(task)
                old_policy = build_policy(old_env)
                set_flat_params(old_policy, theta)
                old_policies.append(old_policy)
            theta = get_flat_params(policy)
        elif resume_from_checkpoint:
            theta = torch.load(resume_from_checkpoint[-1])
            set_flat_params(policy, theta)
        else:
            theta = get_flat_params(policy)
        best_theta = theta.clone()
        all_tasks = []
    elif isinstance(tasks, list) and len(tasks) > 1:  # Multi-env
        all_tasks = tasks
        envs = [make_env(task) for task in all_tasks]
        env_name = "_".join(all_tasks)
        policy = build_policy(envs[0])
        theta = get_flat_params(policy)
        old_tasks = []
        old_policies = []
        resume_from_checkpoint = []
        best_theta = None
    else:  # Single env
        env_name = tasks[0] if isinstance(tasks, list) else tasks
        env = make_env(env_name)
        policy = build_policy(env)
        theta = get_flat_params(policy)
        best_theta = theta.clone()
        all_tasks = []
        old_tasks = []
        old_policies = []
        resume_from_checkpoint = []

    max_steps=None
    total_env_steps = 0
    best_reward = float('-inf')
    
    # Log file setup
    prev_env_name = "None" if not old_tasks else "_".join(old_tasks)
    log_path = f"logs/csv/log_{env_name}_s{sigma}_a{alpha}_n{num_workers}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}_c{prev_env_name}.csv"
    os.makedirs("logs/csv", exist_ok=True)
    if all_tasks:
        log_fields = ['iteration', 'avg_reward', 'std_reward', 'max_reward', 'total_steps', 'env'] + \
                [f'{task}_avg_reward' for task in old_tasks]
    else:
       log_fields = ['iteration', 'avg_reward', 'std_reward', 'max_reward', 'total_steps'] + \
                [f'{task}_avg_reward' for task in old_tasks] 

    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    print("Policy and noise generator built")

    # Spin up persistent actors
    if all_tasks:
        workers = [
            ESWorker.remote(all_tasks, noise)
            for _ in range(num_workers)
        ]
    else:
        workers = [
            ESWorker.remote([env_name], noise)
            for _ in range(num_workers)
        ]


    print("Workers initialized")

    for t in range(iterations): 
        futures = [
            worker.evaluate.remote(theta, sigma, batch_size, max_steps)
            for worker in workers
        ]

        results = ray.get(futures)
        all_indices, all_rewards, all_steps = zip(*results)
        
        all_rewards_flat = [r for batch in all_rewards for r in batch]  # (2N,)
        if rank_function == 'centered':
            shaped = compute_centered_ranks(torch.tensor(all_rewards_flat))  # (2N,)
        elif rank_function == 'weighted':
            shaped = compute_weighted_ranks(torch.tensor(all_rewards_flat))
        elif rank_function == 'fitness':
            shaped = fitness_shaping(torch.tensor(all_rewards_flat))
        elif rank_function == 'z_score':
            shaped = z_score_ranks(torch.tensor(all_rewards_flat))
        else:
            raise ValueError(f"Unknown rank function: {rank_function}")
        
        # Get noises for each antithetic pair
        noises = [
            noise.get(idx, theta.shape[0])
            for batch in all_indices
            for idx in batch
        ]
        noises_tensor = torch.stack([
            val for eps in noises for val in (eps, -eps)
        ])  # shape: (2N, D)


        grad = (shaped.unsqueeze(1) * noises_tensor).mean(dim=0) / sigma  # shape: (D,)

        theta += alpha * grad
        theta = apply_weight_decay(theta, weight_decay)

        
        if all_tasks:
            next_task = all_tasks[(t + 1) % len(all_tasks)]
            curr_env = envs[t%len(envs)]
            ray.get([worker.set_env.remote(theta, curr_env, next_task) for worker in workers[:-1]])
            theta = ray.get(workers[-1].set_env.remote(theta, curr_env, next_task))
        else:
            ray.get([worker.set_policy.remote(theta) for worker in workers])
            curr_env = env


        # Adaptive max_steps (capped at 2x mean_length)
        episode_lengths = [l for batch in all_steps for l in batch]  # (2N, )
        if adaptive_max_steps:
            mean_length = sum(episode_lengths) / len(episode_lengths)
            max_steps = int(2 * mean_length)
        total_env_steps += sum(episode_lengths)

        avg_reward = torch.tensor(all_rewards_flat).mean()
        std_reward = torch.tensor(all_rewards_flat).std()
        max_reward = max(all_rewards_flat)
        print(f"Iter {t+1:03d}: avg raw reward = {avg_reward:.2f} ± {std_reward:.2f}, max = {max_reward:.2f}, max_steps = {max_steps}, total steps = {total_env_steps}, env = {curr_env.spec.id}")

        # Keep track of the best policy
        if avg_reward > best_reward and not all_tasks:
            best_reward = avg_reward
            best_theta = theta.clone()

        if all_tasks:
            log_row = {
                'iteration': t + 1,
                'avg_reward': avg_reward.item(),
                'std_reward': std_reward.item(),
                'max_reward': max_reward,
                'total_steps': total_env_steps,
                'env': curr_env.spec.id
            }
        else:
            log_row = {
                'iteration': t + 1,
                'avg_reward': avg_reward.item(),
                'std_reward': std_reward.item(),
                'max_reward': max_reward,
                'total_steps': total_env_steps
            }

        # evaluate old tasks if provided
        if old_tasks and (t == 0 or (t+1) % 10 == 0):
            for i, old_policy in enumerate(old_policies):
                # extract_and_transfer_hidden(theta, env, old_policy)
                old_rewards = []
                old_env = make_env(old_tasks[i])
                for j in range(5):  # Evaluate each old task 5 times
                    old_reward, _ = rollout(old_policy, old_env)
                    old_rewards.append(old_reward)

                old_avg = sum(old_rewards) / len(old_rewards)
                log_row[f'{old_tasks[i]}_avg_reward'] = old_avg
                print(f"Old task {old_tasks[i]}: avg reward = {old_avg:.2f}")

        # Log the results
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)

        # Checkpointing
        if checkpoint_interval != 0 and (t + 1) % checkpoint_interval == 0:
            print(f"Checkpointing at iteration {t + 1}")
            torch.save(theta, f"chkpts/policy_{env_name}_s{sigma}_a{alpha}_n{num_workers}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}_c{prev_env_name}_tes{total_env_steps}_rew{avg_reward}.pt")

    if all_tasks:
        best_theta = theta.clone()  # For all tasks, we keep the latest theta as best

    set_flat_params(policy, best_theta)
    torch.save(best_theta, f"chkpts/best_policy_{env_name}_s{sigma}_a{alpha}_n{num_workers}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}_c{prev_env_name}_tes{total_env_steps}_rew{avg_reward}.pt")
