import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, output_activations, shared_output=False):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.output_activations = output_activations
        self.shared_output = shared_output
        self.max_output_dim = max(output_dims) if self.shared_output else None

        self.input_hidden = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0]) for input_dim in input_dims])
        self.hidden = nn.Sequential(*[nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])
        self.output = nn.ModuleList([nn.Linear(hidden_dims[-1], output_dim) for output_dim in output_dims]) if not self.shared_output else nn.ModuleList([nn.Linear(hidden_dims[-1], self.max_output_dim)])
        #possibile bug con output_activations, nel caso in cui le funzioni siano diverse per task (ma in quel caso ci sarebbe da cambiare un po' tutto)


    def forward(self, x, task_id):
        task_id_output = task_id if not self.shared_output else 0
        x = torch.tanh((self.input_hidden[task_id])(x))
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        y = self.output_activations[task_id_output]((self.output[task_id_output])(x))
        d = self.output_dims[task_id]
        return y[..., :d] if self.shared_output else y

def substitute_task(policy, task_id, new_input_dim, new_output_dim, new_output_activation):
    """
    Substitute the task-specific layers with new dimensions and activation.
    """
    policy.input_dims[task_id] = new_input_dim
    policy.output_dims[task_id] = new_output_dim
    policy.output_activations[task_id] = new_output_activation
    
    policy.input_hidden[task_id] = nn.Linear(new_input_dim, policy.hidden_dims[0])
    if not policy.shared_output:
        policy.output[task_id] = nn.Linear(policy.hidden_dims[-1], new_output_dim)
        policy.output_activations[task_id] = new_output_activation

def get_flat_params(model, task_id):
    """
    Get flattened parameters of the model for a specific task.
    """
    task_id_output = task_id if not model.shared_output else 0
    return torch.cat([p.data.view(-1) for p in model.input_hidden[task_id].parameters()] +
                     [p.data.view(-1) for p in model.hidden.parameters()] +
                     [p.data.view(-1) for p in model.output[task_id_output].parameters()])

def set_flat_params(model, flat_params, task_id):
    """
    Set flattened parameters of the model for a specific task.
    """
    task_id_output = task_id if not model.shared_output else 0
    pointer = 0
    for param in model.input_hidden[task_id].parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer + numel].view(param.size()))
        pointer += numel

    for param in model.hidden.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer + numel].view(param.size()))
        pointer += numel

    for param in model.output[task_id_output].parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer + numel].view(param.size()))
        pointer += numel

