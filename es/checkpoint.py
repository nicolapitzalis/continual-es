import os

import numpy as np
import torch

from es.policy import Policy, add_task


REQUIRED_CHECKPOINT_KEYS = {
    "env_names",
    "input_dims",
    "hidden_dims",
    "output_dims",
    "output_activation",
    "shared_output",
    "action_bins",
    "adaptive_max_steps",
    "adaptive_max_steps_start",
    "adaptive_max_steps_threshold",
    "adaptive_max_steps_ratio",
    "max_steps_by_task",
    "normalize_observations",
    "obs_clip",
    "observation_stats",
    "state_dict",
}


def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Checkpoint is missing required fields: {missing_names}")
    return checkpoint


def validate_checkpoint_compatibility(checkpoint, config):
    if config.action_bins != checkpoint["action_bins"]:
        raise ValueError(
            f"Checkpoint uses action_bins={checkpoint['action_bins']}, "
            f"but action_bins={config.action_bins} was requested"
        )
    if config.normalize_observations != checkpoint["normalize_observations"]:
        raise ValueError(
            "Checkpoint and requested observation-normalization settings differ"
        )
    if config.normalize_observations and not np.isclose(
        config.obs_clip,
        checkpoint["obs_clip"],
    ):
        raise ValueError(
            f"Checkpoint uses obs_clip={checkpoint['obs_clip']}, "
            f"but obs_clip={config.obs_clip} was requested"
        )
    if config.adaptive_max_steps != checkpoint["adaptive_max_steps"]:
        raise ValueError(
            "Checkpoint and requested adaptive max-step settings differ"
        )

    if config.adaptive_max_steps:
        saved_config = (
            checkpoint["adaptive_max_steps_start"],
            checkpoint["adaptive_max_steps_threshold"],
            checkpoint["adaptive_max_steps_ratio"],
        )
        requested_config = (
            config.adaptive_max_steps_start,
            config.adaptive_max_steps_threshold,
            config.adaptive_max_steps_ratio,
        )
        if not all(
            np.isclose(saved, requested)
            for saved, requested in zip(saved_config, requested_config)
        ):
            raise ValueError(
                "Adaptive max-step configuration does not match the checkpoint: "
                f"checkpoint={saved_config}, requested={requested_config}"
            )


def prepare_continual_tasks(checkpoint, new_env_names):
    if len(new_env_names) != 1:
        raise ValueError(
            "Continual training expects exactly one new environment; "
            "previous environments are loaded from the checkpoint"
        )
    previous_env_names = list(checkpoint["env_names"])
    new_env_name = new_env_names[0]
    if new_env_name in previous_env_names:
        raise ValueError(
            f"Environment {new_env_name!r} is already present in the checkpoint; "
            "continual training expects a new task"
        )
    return previous_env_names + [new_env_name], len(previous_env_names)


def restore_policy(
    checkpoint,
    input_dims,
    output_dims,
    output_activations,
    current_task_id,
):
    policy = Policy(
        checkpoint["input_dims"],
        checkpoint["hidden_dims"],
        checkpoint["output_dims"],
        checkpoint["output_activation"],
        checkpoint["shared_output"],
    )
    policy.load_state_dict(checkpoint["state_dict"])

    if len(policy.input_dims) != current_task_id:
        raise ValueError(
            f"Checkpoint contains {len(policy.input_dims)} policy tasks but "
            f"{current_task_id} environment names"
        )
    if (
        input_dims[:current_task_id] != policy.input_dims
        or output_dims[:current_task_id] != policy.output_dims
    ):
        raise ValueError(
            "The saved environments no longer match the policy dimensions "
            "in the checkpoint"
        )

    add_task(
        policy,
        input_dims[current_task_id],
        output_dims[current_task_id],
        output_activations[current_task_id],
    )
    return policy


def build_checkpoint_payload(
    *,
    env_names,
    policy,
    config,
    observation_stats,
    max_steps_by_task,
):
    return {
        "env_names": list(env_names),
        "input_dims": policy.input_dims,
        "hidden_dims": policy.hidden_dims,
        "output_dims": policy.output_dims,
        "output_activation": policy.output_activations,
        "shared_output": policy.shared_output,
        "action_bins": config.action_bins,
        "adaptive_max_steps": config.adaptive_max_steps,
        "adaptive_max_steps_start": config.adaptive_max_steps_start,
        "adaptive_max_steps_threshold": config.adaptive_max_steps_threshold,
        "adaptive_max_steps_ratio": config.adaptive_max_steps_ratio,
        "max_steps_by_task": list(max_steps_by_task),
        "normalize_observations": config.normalize_observations,
        "obs_clip": config.obs_clip,
        "observation_stats": observation_stats,
        "state_dict": policy.state_dict(),
    }


def save_checkpoint(path, **payload_arguments):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(build_checkpoint_payload(**payload_arguments), path)

