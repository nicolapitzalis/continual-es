import csv
import os


def build_run_name(env_names, current_task_id, multi_task, config):
    if multi_task:
        environment_label = "_".join(env_names)
        run_name = (
            f"log_{environment_label}_s{config.sigma}_a{config.alpha}"
            f"_i{config.iterations}_d{config.num_directions}"
            f"_w{config.weight_decay}_{config.rank_function}"
            f"_ams{config.adaptive_max_steps}"
        )
    else:
        current_env = env_names[current_task_id]
        run_name = (
            f"log_{current_env}_s{config.sigma}_a{config.alpha}"
            f"_i{config.iterations}_d{config.num_directions}"
            f"_w{config.weight_decay}_{config.rank_function}"
            f"_ams{config.adaptive_max_steps}"
        )
        if current_task_id > 0:
            previous_envs = "_".join(env_names[:current_task_id])
            run_name += (
                f"_{previous_envs}_replay{config.replay_directions}"
                f"_every{config.replay_interval}"
            )

    if config.adaptive_max_steps:
        run_name += (
            f"_start{config.adaptive_max_steps_start}"
            f"_threshold{config.adaptive_max_steps_threshold}"
            f"_ratio{config.adaptive_max_steps_ratio}"
        )
    if config.shared_output:
        run_name += "_shared_output"
    if config.action_bins > 1:
        run_name += f"_bins{config.action_bins}"
    if config.normalize_observations:
        run_name += "_obsnorm"
    if config.frozen_hidden and not multi_task:
        run_name += "_frozen_hidden"
    run_name += f"_seed{config.seed}"
    return run_name


def log_fields(env_names, current_task_id, multi_task, include_fwt):
    fields = [
        "iteration",
        "avg_reward",
        "std_reward",
        "max_reward",
        "max_steps",
        "total_steps",
        "task",
    ]
    if multi_task:
        fields.extend(f"{task}_eval_curr_policy" for task in env_names)
    else:
        fields.append("eval_curr_policy")
        fields.extend(
            f"{task}_avg_reward" for task in env_names[:current_task_id]
        )
    if include_fwt:
        fields.extend(
            f"{task}_fwt_before"
            for task in env_names[current_task_id + 1 :]
        )
    return fields


class CsvLogger:
    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=fields).writeheader()

    def write(self, row):
        with open(self.path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writerow(row)
