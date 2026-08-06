import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
        output_activations,
        shared_output=False,
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.hidden_dims = list(hidden_dims)
        self.output_dims = list(output_dims)
        self.output_activations = list(output_activations)
        self.shared_output = shared_output
        self.max_output_dim = max(output_dims) if shared_output else None

        self.input_hidden = nn.ModuleList(
            nn.Linear(input_dim, hidden_dims[0]) for input_dim in input_dims
        )
        self.hidden = nn.Sequential(
            *(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            )
        )
        if shared_output:
            self.output = nn.ModuleList(
                [nn.Linear(hidden_dims[-1], self.max_output_dim)]
            )
        else:
            self.output = nn.ModuleList(
                nn.Linear(hidden_dims[-1], output_dim)
                for output_dim in output_dims
            )

    def forward(self, x, task_id):
        output_id = 0 if self.shared_output else task_id
        x = torch.tanh(self.input_hidden[task_id](x))
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        output = self.output_activations[task_id](self.output[output_id](x))
        task_output_dim = self.output_dims[task_id]
        return output[..., :task_output_dim] if self.shared_output else output


def add_task(policy, input_dim, output_dim, output_activation):
    """Append task-specific policy components for a newly observed task."""
    reference_param = next(policy.parameters())
    device = reference_param.device
    dtype = reference_param.dtype

    input_layer = nn.Linear(input_dim, policy.hidden_dims[0]).to(
        device=device,
        dtype=dtype,
    )
    policy.input_hidden.append(input_layer)
    policy.input_dims.append(input_dim)
    policy.output_dims.append(output_dim)
    policy.output_activations.append(output_activation)

    if not policy.shared_output:
        output_layer = nn.Linear(policy.hidden_dims[-1], output_dim).to(
            device=device,
            dtype=dtype,
        )
        policy.output.append(output_layer)
        return

    if output_dim <= policy.max_output_dim:
        return

    old_output = policy.output[0]
    expanded_output = nn.Linear(policy.hidden_dims[-1], output_dim).to(
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        expanded_output.weight[: policy.max_output_dim].copy_(old_output.weight)
        expanded_output.bias[: policy.max_output_dim].copy_(old_output.bias)

    policy.output[0] = expanded_output
    policy.max_output_dim = output_dim


def get_flat_params(model, task_id):
    output_id = 0 if model.shared_output else task_id
    parameters = (
        list(model.input_hidden[task_id].parameters())
        + list(model.hidden.parameters())
        + list(model.output[output_id].parameters())
    )
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def set_flat_params(model, flat_params, task_id):
    output_id = 0 if model.shared_output else task_id
    parameters = (
        list(model.input_hidden[task_id].parameters())
        + list(model.hidden.parameters())
        + list(model.output[output_id].parameters())
    )

    pointer = 0
    with torch.no_grad():
        for parameter in parameters:
            numel = parameter.numel()
            parameter.copy_(
                flat_params[pointer : pointer + numel].view_as(parameter)
            )
            pointer += numel
    if pointer != flat_params.numel():
        raise ValueError(
            f"Flat parameter length mismatch: consumed {pointer}, "
            f"received {flat_params.numel()}"
        )

