import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """
    Modular policy network with easily swappable input and output layers.
    """
    def __init__(self, input_layer, backbone, output_layer, output_activation):
        super().__init__()
        self.input_layer = input_layer
        self.backbone = backbone
        self.output_layer = output_layer
        self.output_activation = output_activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.backbone(x)
        return self.output_activation(self.output_layer(x))

def make_mlp(input_dim, hidden_sizes, activation=nn.Tanh):
    layers = []
    dims = [input_dim] + hidden_sizes
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    return nn.Sequential(*layers)

def build_policy(env, hidden_sizes=[64, 64]):
    """
    Build a modular Policy network for the given environment.
    """
    obs_dim = env.observation_space.shape[0]
    act_space = env.action_space

    if hasattr(act_space, 'n'):
        act_dim = act_space.n
        output_activation = lambda x: F.softmax(x, dim=-1)
    else:
        act_dim = act_space.shape[0]
        output_activation = torch.tanh

    input_layer = nn.Identity()
    backbone = make_mlp(obs_dim, hidden_sizes)
    output_layer = nn.Linear(hidden_sizes[-1], act_dim)
    return Policy(input_layer, backbone, output_layer, output_activation)


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer + numel].view(param.size()))
        pointer += numel

def extract_and_transfer_hidden(theta, source_env, target_policy, hidden_sizes=[64, 64]):
    """
    Loads flat parameters into a temp policy for `source_env`,
    then copies hidden layer weights into `target_policy`.
    Skips input and output layers, only transfers backbone weights.
    """
    temp_policy = build_policy(source_env, hidden_sizes)
    set_flat_params(temp_policy, theta)
    # Copy hidden layers (excluding input/output) to target
    src_layers = [m for m in temp_policy.backbone if isinstance(m, nn.Linear)]
    tgt_layers = [m for m in target_policy.backbone if isinstance(m, nn.Linear)]
    # Skip input and output layers
    for src, tgt in zip(src_layers[1:-1], tgt_layers[1:-1]):
        tgt.weight.data.copy_(src.weight.data)
        tgt.bias.data.copy_(src.bias.data)
