import torch


def fitness_shaping(values):
    count = values.shape[0]
    ranks = torch.argsort(torch.argsort(-values)).float() + 1
    log_base = torch.log(
        torch.tensor(
            count / 2 + 1.0,
            device=ranks.device,
            dtype=ranks.dtype,
        )
    )
    utilities = torch.clamp(log_base - torch.log(ranks), min=0.0)
    utilities /= utilities.sum()
    utilities -= 1.0 / count
    return utilities


def z_score_ranks(values):
    ranks = torch.argsort(torch.argsort(values)).float()
    return (ranks - ranks.mean()) / (ranks.std() + 1e-8)


def compute_centered_ranks(values):
    ranks = torch.argsort(torch.argsort(values)).float()
    return ranks / (len(values) - 1) - 0.5


def compute_weighted_ranks(values):
    ranks = torch.argsort(torch.argsort(-values)).float()
    return (len(values) - 1 - ranks) / (len(values) - 1)


def process_returns(returns, rank_function):
    values = torch.as_tensor(returns, dtype=torch.float32)
    processors = {
        "centered": compute_centered_ranks,
        "weighted": compute_weighted_ranks,
        "fitness_shaping": fitness_shaping,
        "z_score": z_score_ranks,
        "none": lambda raw_values: raw_values,
    }
    try:
        return processors[rank_function](values)
    except KeyError as error:
        raise ValueError(f"Unknown rank function: {rank_function}") from error


def compute_gradient(results, theta_dim, rank_function, noise, sigma):
    all_indices = [result[0] for result in results]
    all_rewards = [result[1] for result in results]
    all_steps = [result[2] for result in results]
    flat_rewards = [reward for batch in all_rewards for reward in batch]
    processed_returns = process_returns(flat_rewards, rank_function)

    perturbations = [
        noise.get(index, theta_dim)
        for batch in all_indices
        for index in batch
    ]
    antithetic_noise = torch.stack(
        [value for epsilon in perturbations for value in (epsilon, -epsilon)]
    )
    gradient = (
        processed_returns.unsqueeze(1) * antithetic_noise
    ).mean(dim=0) / sigma
    return gradient, flat_rewards, all_steps


def _parameter_sizes(model, task_id):
    output_id = 0 if model.shared_output else task_id
    input_size = sum(
        parameter.numel()
        for parameter in model.input_hidden[task_id].parameters()
    )
    hidden_size = sum(parameter.numel() for parameter in model.hidden.parameters())
    output_size = sum(
        parameter.numel() for parameter in model.output[output_id].parameters()
    )
    return input_size, hidden_size, output_size


def _parameter_slices(model, task_id):
    input_size, hidden_size, output_size = _parameter_sizes(model, task_id)
    input_slice = slice(0, input_size)
    hidden_slice = slice(input_size, input_size + hidden_size)
    output_slice = slice(
        input_size + hidden_size,
        input_size + hidden_size + output_size,
    )
    return input_slice, hidden_slice, output_slice


def repack_replay_gradient(gradient, model, from_task, to_task):
    input_size, hidden_size, output_size = _parameter_sizes(model, to_task)
    packed = torch.zeros(
        input_size + hidden_size + output_size,
        dtype=gradient.dtype,
        device=gradient.device,
    )

    _, from_hidden, from_output = _parameter_slices(model, from_task)
    _, to_hidden, to_output = _parameter_slices(model, to_task)
    packed[to_hidden] = gradient[from_hidden]
    if model.shared_output:
        packed[to_output] = gradient[from_output]
    return packed


def apply_es_gradient(policy, optimizer, gradient, task_id, frozen_hidden):
    optimizer.zero_grad(set_to_none=True)
    output_id = 0 if policy.shared_output else task_id
    parameter_groups = (
        (False, policy.input_hidden[task_id].parameters()),
        (frozen_hidden, policy.hidden.parameters()),
        (False, policy.output[output_id].parameters()),
    )

    pointer = 0
    for skip_update, parameters in parameter_groups:
        for parameter in parameters:
            numel = parameter.numel()
            parameter_gradient = gradient[pointer : pointer + numel]
            pointer += numel
            if not skip_update:
                parameter.grad = -parameter_gradient.view_as(parameter)

    if pointer != gradient.numel():
        raise ValueError(
            f"Gradient length mismatch: consumed {pointer}, "
            f"received {gradient.numel()}"
        )
    optimizer.step()

