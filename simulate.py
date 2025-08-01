import torch
from environments.env_utils import make_env
from core.policy import get_flat_params, set_flat_params, build_policy


env = make_env("Hopper-v5", render_mode="rgb_array")
policy = build_policy(env)
theta = torch.load("/home/n.pitzalis/es/chkpts/policy_Hopper-v5_s0.1_a0.05_n12_b32_w0.0_centered_amsTrue_daFalse_tes633022077_rew1538.7653101614223.pt")
set_flat_params(policy, theta)

obs, _ = env.reset()
done = False
frames = []
total_reward = 0.0

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = policy(obs_tensor)
    action = action.squeeze().detach().numpy()
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

env.close()
print(f"Total reward: {total_reward}")