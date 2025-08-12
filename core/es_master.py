import ray
import torch
import csv
import os
from core.policy import Policy, get_flat_params, set_flat_params, substitute_task
from distributed.worker import ESWorker
from core.noise_generator import NoiseGenerator
from functional.utils import compute_centered_ranks, compute_weighted_ranks, fitness_shaping, z_score_ranks
from core.updates import apply_weight_decay
from core.rollout import rollout
from environments.env_utils import make_env, exctract_envs_info, eval_policy

def train(
    envs=[],
    to_train=0,         # if to_train >= len(envs) then is is multi-task case
    sigma=0.1,
    alpha=0.05,
    hidden_dims=[64, 64],
    output_activation=torch.tanh,
    iterations=1000,
    num_workers=12,
    batch_size=32,
    weight_decay=0.005,
    rank_function='centered',
    ray_address=None,
    adaptive_max_steps=True,
    checkpoint_interval=100,
    checkpoint=None,
    shared_output=False
):
    ray.init(address=ray_address)
    print("Ray initialized")
    noise = NoiseGenerator(seed=42)
    max_steps = None
    total_env_steps = 0
    best_reward = float('-inf')
    best_theta = None
    if to_train >= len(envs):
        multi_task_case = True
        to_train = 0
    else:
        multi_task_case = False
    real_envs = [make_env(env_name) for env_name in envs]


    # Files setup
    # ------------------------------------------------
    if multi_task_case:
        log_name = "_".join(envs)
        file_name = f"log_{log_name}_s{sigma}_a{alpha}_i{iterations}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}"
        if shared_output:
            file_name += "_shared_output"
        log_path = f"logs/csv/{file_name}.csv"
    else:
        if to_train > 0:
            prev_envs = "_".join(envs[:to_train])
            file_name = f"log_{envs[to_train]}_s{sigma}_a{alpha}_i{iterations}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}_{prev_envs}"
        else:
            file_name = f"log_{envs[to_train]}_s{sigma}_a{alpha}_i{iterations}_b{batch_size}_w{weight_decay}_{rank_function}_ams{adaptive_max_steps}"
        if shared_output:
            file_name += "_shared_output"
        log_path = f"logs/csv/{file_name}.csv"
    os.makedirs("logs/csv", exist_ok=True)
    if multi_task_case:
        log_fields = ['iteration', 'avg_reward', 'std_reward', 'max_reward', 'total_steps', 'env'] + \
                [f'{task}_eval_curr_policy' for task in envs]
    else:
       log_fields = ['iteration', 'avg_reward', 'std_reward', 'max_reward', 'total_steps', 'eval_curr_policy'] + \
                [f'{task}_avg_reward' for task in envs[:to_train]]

    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_fields)
        writer.writeheader()
    # -------------------------------------------------

    # Build policy
    # ------------------------------------------------
    if checkpoint:      # continual case
        model_checkpoint = torch.load(checkpoint, weights_only=False)
        mc_shared_output = model_checkpoint.get('shared_output', False)

        policy = Policy(
            model_checkpoint['input_dims'],
            model_checkpoint['hidden_dims'],
            model_checkpoint['output_dims'],
            model_checkpoint['output_activation'],
            mc_shared_output
        )
        policy.load_state_dict(model_checkpoint['state_dict'])

        # when using a checkpoint, tasks need to be in the same order as in the checkpoint
        input_dims, output_dims, output_activation = exctract_envs_info(real_envs)
        if input_dims != policy.input_dims:
            print(f"Warning: input dimensions mismatch. Possible continual case with different task ordering. Substituting task-specific layers.")
            for i in range(len(input_dims)):
                if input_dims[i] != policy.input_dims[i]:
                    substitute_task(policy, i, input_dims[i], output_dims[i], output_activation[i])

    else:               # single/multi task case
        input_dims, output_dims, output_activation = exctract_envs_info(real_envs)
        policy = Policy(input_dims, hidden_dims, output_dims, output_activation, shared_output=shared_output)
    
    theta = get_flat_params(policy, to_train)
    print(f"Policy initialized for task {envs[to_train]} with input dims {policy.input_dims}, hidden dims {hidden_dims}, output dims {policy.output_dims}, theta shape {theta.shape}")
    # ------------------------------------------------
    workers = [
        ESWorker.remote(env_names=envs, noise=noise, hidden_sizes=hidden_dims, shared_output=shared_output)
        for _ in range(num_workers)
    ]
    print(f"Workers initialized: {len(workers)}")

    for t in range(iterations):
        # Training phase
        # ------------------------------------------------
        futures = [
            worker.evaluate.remote(
                to_train, theta, sigma, batch_size, max_steps
            )
            for worker in workers
        ]

        results = ray.get(futures)
        all_indices, all_rewards, all_steps = zip(*results)
        all_rewards_flat = [r for batch in all_rewards for r in batch]
        if rank_function == 'centered':
            ranks = compute_centered_ranks(torch.tensor(all_rewards_flat))
        elif rank_function == 'weighted':
            ranks = compute_weighted_ranks(torch.tensor(all_rewards_flat))
        elif rank_function == 'fitness_shaping':
            ranks = fitness_shaping(torch.tensor(all_rewards_flat))
        elif rank_function == 'z_score':
            ranks = z_score_ranks(torch.tensor(all_rewards_flat))
        else:
            raise ValueError(f"Unknown rank function: {rank_function}")

        # reconstruct noises for each antithetic pair
        noises = [
            noise.get(idx, theta.shape[0])
            for batch_size in all_indices
            for idx in batch_size
        ]
        noises_tensor = torch.stack([
            val for eps in noises for val in (eps, -eps)
        ])

        # compute gradient and update theta
        grad = (ranks.unsqueeze(1) * noises_tensor).mean(dim=0) / sigma
        theta += alpha * grad
        theta = apply_weight_decay(theta, weight_decay)
        ray.get([worker.set_policy.remote(theta, to_train) for worker in workers])
        
        avg_reward = torch.tensor(all_rewards_flat).mean()
        std_reward = torch.tensor(all_rewards_flat).std()
        max_reward = max(all_rewards_flat)
        # ------------------------------------------------

        # Logging
        # ------------------------------------------------
        log_row = {
            'iteration': t + 1,
            'avg_reward': avg_reward.item(),
            'std_reward': std_reward.item(),
            'max_reward': max_reward,
            'total_steps': total_env_steps,
        }
        if multi_task_case:
            log_row['env'] = envs[to_train]

        print(f"Iter {t+1:03d}: avg raw reward = {avg_reward:.2f} ± {std_reward:.2f}, max = {max_reward:.2f}, max_steps = {max_steps}, total steps = {total_env_steps}, env = {envs[to_train]}")
        # -------------------------------------------------

        # Evaluation phase
        # ------------------------------------------------
        if t == 0 or (t +1) % 10 == 0:
            # evaluate current env (clean policy)
            set_flat_params(policy, theta, to_train)        # updates policy[to_train] (hiddens are updated)
            
            range_envs = range(len(envs)) if multi_task_case else [to_train]
            for i in range_envs:
                eval_policy_res = eval_policy(policy, real_envs, i, max_steps=max_steps, iterations=10)
                if multi_task_case:
                    log_row[f'{envs[i]}_eval_curr_policy'] = eval_policy_res
                else:
                    log_row['eval_curr_policy'] = eval_policy_res
                print(f"Eval current policy on env {envs[i]}: avg reward = {eval_policy_res:.2f}")

            # evaluate previous envs (if continual case)
            if checkpoint and to_train > 0:
                for i in range(to_train):
                    old_env_rew = eval_policy(policy, real_envs, i, max_steps=max_steps, iterations=10)
                    log_row[f'{envs[i]}_avg_reward'] = old_env_rew
                    print(f"Eval old env {envs[i]}: avg reward = {old_env_rew:.2f}")
        # ------------------------------------------------

        # Save log
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_fields)
            writer.writerow(log_row)

        if multi_task_case:
            set_flat_params(policy, theta, to_train)  # updates policy[to_train]
            to_train = (to_train + 1) % len(envs)           # switch to next task
            theta = get_flat_params(policy, to_train)  # get new theta for the next task
        
        # Adaptive max steps
        # ------------------------------------------------
        episode_lengths = [l for batch in all_steps for l in batch]
        if adaptive_max_steps and not multi_task_case:
            mean_length = sum(episode_lengths) / len(episode_lengths)
            max_steps = int(2 * mean_length)
        total_env_steps += len(episode_lengths)
        # ------------------------------------------------

        # save best model (not in the multi-task case)
        if avg_reward > best_reward and not multi_task_case:
            best_reward = avg_reward
            best_theta = theta.clone()

        # Save checkpoint
        # ------------------------------------------------
        if checkpoint_interval != 0 and (t + 1) % checkpoint_interval == 0:
            torch.save({
                'input_dims': policy.input_dims,
                'hidden_dims': hidden_dims,
                'output_dims': policy.output_dims,
                'output_activation': output_activation,
                'shared_output': shared_output,
                'state_dict': policy.state_dict(),
            }, f"chkpts/checkpoint_{file_name}_at{t+1}.pth")
        # ------------------------------------------------
    
    if multi_task_case:
        best_theta = theta.clone()  # in multi-task case, we return the last theta
    
    # Save the best policy
    set_flat_params(policy, best_theta, to_train)
    print(f"Training completed. Best theta saved for task {envs[to_train]}.")
    torch.save({
        'input_dims': policy.input_dims,
        'hidden_dims': hidden_dims,
        'output_dims': policy.output_dims,
        'output_activation': output_activation,
        'shared_output': shared_output,
        'state_dict': policy.state_dict(),
    }, f"chkpts/best_policy_{file_name}.pth")

       
