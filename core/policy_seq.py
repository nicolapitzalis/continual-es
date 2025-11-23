import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, discrete_action=False):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_sizes[-1], output_dim)
        self.discrete_action = discrete_action

    def forward(self, x):
        x = self.backbone(x)
        if self.discrete_action:
            return F.softmax(self.output(x), dim=-1)
        else:
            return torch.tanh(self.output(x))

class DiscretizedPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes, bins=10):
        super().__init__()
        self.bins = bins
        self.action_dim = action_dim
        self.discrete_action = False  # continuous env, but discretized output

        layers = []
        dims = [input_dim] + hidden_sizes
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        # Output: logits for each bin of each action dimension
        self.output = nn.Linear(hidden_sizes[-1], action_dim * bins)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.output(x)
        logits = logits.view(-1, self.action_dim, self.bins)  # shape: [batch, dim, bins]
        probs = F.softmax(logits, dim=-1)
        # expected value from bin centers ∈ [−1, 1]
        bin_centers = torch.linspace(-1, 1, self.bins, device=probs.device)
        actions = (probs * bin_centers).sum(dim=-1)  # shape: [batch, dim]
        return actions


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer + numel].view(param.size()))
        pointer += numel

def build_policy(env, hidden_sizes=[64, 64], discretized=False, bins=10):
    obs_dim = env.observation_space.shape[0]
    act_space = env.action_space

    # Running on continuous env with discretized actions
    if discretized:
        return DiscretizedPolicy(obs_dim, act_space.shape[0], hidden_sizes, bins)

    if hasattr(act_space, 'n'):
        discrete_action = True
        act_dim = act_space.n
    else:
        act_dim = act_space.shape[0]
        discrete_action = False
    return Policy(obs_dim, act_dim, hidden_sizes, discrete_action)

def extract_and_transfer_hidden(theta, source_env, target_policy, hidden_sizes=[64, 64], discretized=False):
    """
    Loads flat parameters into a temp policy for `source_env`,
    then copies hidden layer weights into `target_policy`.
    """
    # Build temp policy to unpack theta
    temp_policy = build_policy(source_env, hidden_sizes, discretized)
    set_flat_params(temp_policy, theta)

    # Copy hidden layers (excluding input/output) to target
    src_layers = temp_policy.backbone[2:-1:2]  # Linear layers only, skipping input
    tgt_layers = target_policy.backbone[2:-1:2]

    for src, tgt in zip(src_layers, tgt_layers):
        tgt.weight.data.copy_(src.weight.data)
        tgt.bias.data.copy_(src.bias.data)
