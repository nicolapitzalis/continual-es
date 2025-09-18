import torch
from functional.utils import compute_centered_ranks

def compute_gradient_seq(rewards, noises, sigma, rank=True):
    dim = noises.shape[0]
    rewards = rewards.view(dim, 2)
    paired = rewards[:, 0] - rewards[:, 1]
    if rank:
        paired = compute_centered_ranks(paired)
    return (paired @ noises) / (dim * sigma)

def apply_weight_decay(theta, weight_decay):
    return theta * (1.0 - weight_decay)

def compute_gradient(res, theta_dim, rank_function, noise, sigma):
    all_indices, all_rewards, all_steps = zip(*res)
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
        noise.get(idx, theta_dim)
        for batch in all_indices
        for idx in batch
    ]
    noises_tensor = torch.stack([
        val for eps in noises for val in (eps, -eps)
    ])

    # compute gradient and update theta
    grad = (ranks.unsqueeze(1) * noises_tensor).mean(dim=0) / sigma
    return grad, all_rewards_flat, all_steps


# --- helpers: param sizes & slices per task ---
def param_sizes(model, task_id):
    Lin  = sum(p.numel() for p in model.input_hidden[task_id].parameters())
    Lhid = sum(p.numel() for p in model.hidden.parameters())
    Lout = sum(p.numel() for p in model.output[0].parameters())  # shared output
    return Lin, Lhid, Lout

def slices_for(model, task_id):
    Lin, Lhid, Lout = param_sizes(model, task_id)
    s_in  = slice(0, Lin)
    s_hid = slice(Lin, Lin + Lhid)
    s_out = slice(Lin + Lhid, Lin + Lhid + Lout)
    return s_in, s_hid, s_out

def repack_replay_grad(grad_from, model, from_task, to_task):
    # Build zero target grad with the *current* task's layout
    Lin_to, Lhid_to, Lout_to = param_sizes(model, to_task)
    target_len = Lin_to + Lhid_to + Lout_to
    g = torch.zeros(target_len, dtype=grad_from.dtype, device=grad_from.device)

    # Copy only shared parts (hidden + shared output)
    s_in_from, s_hid_from, s_out_from = slices_for(model, from_task)
    s_in_to,   s_hid_to,   s_out_to   = slices_for(model, to_task)
    g[s_hid_to] = grad_from[s_hid_from]
    g[s_out_to] = grad_from[s_out_from]
    # input slice remains zero (task-specific)
    return g
