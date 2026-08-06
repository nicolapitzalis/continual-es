from dataclasses import dataclass


RANK_FUNCTIONS = (
    "centered",
    "weighted",
    "z_score",
    "fitness_shaping",
    "none",
)


@dataclass(frozen=True)
class TrainingConfig:
    sigma: float = 0.1
    alpha: float = 0.05
    hidden_dims: tuple[int, ...] = (64, 64)
    iterations: int = 1000
    num_workers: int = 12
    num_directions: int = 384
    action_bins: int = 1
    weight_decay: float = 0.005
    rank_function: str = "centered"
    adaptive_max_steps: bool = False
    adaptive_max_steps_start: int = 100
    adaptive_max_steps_threshold: float = 0.7
    adaptive_max_steps_ratio: float = 2.0
    normalize_observations: bool = False
    obs_clip: float = 5.0
    checkpoint_interval: int = 100
    shared_output: bool = False
    replay_directions: int = 0
    replay_interval: int = 1
    replay_weight: float = 1.0
    frozen_hidden: bool = False
    fwt: bool = False
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        self.validate()

    def validate(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.action_bins < 1:
            raise ValueError("action_bins must be at least 1")
        if self.num_directions <= 0:
            raise ValueError("num_directions must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.rank_function not in RANK_FUNCTIONS:
            raise ValueError(f"Unknown rank function: {self.rank_function}")
        if self.obs_clip <= 0:
            raise ValueError("obs_clip must be positive")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.replay_directions < 0:
            raise ValueError("replay_directions must be non-negative")
        if self.replay_interval <= 0:
            raise ValueError("replay_interval must be positive")
        if self.replay_weight < 0:
            raise ValueError("replay_weight must be non-negative")

        if self.adaptive_max_steps:
            if self.adaptive_max_steps_start <= 0:
                raise ValueError("adaptive_max_steps_start must be positive")
            if not 0 < self.adaptive_max_steps_threshold <= 1:
                raise ValueError("adaptive_max_steps_threshold must be in (0, 1]")
            if self.adaptive_max_steps_ratio <= 1:
                raise ValueError("adaptive_max_steps_ratio must be greater than 1")
