import gymnasium as gym

def make_env(env_name, seed=None, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
    