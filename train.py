"""Entry point for rollout collection and PPO training."""

import argparse
import pickle
import shutil
from pathlib import Path

from paths import disconnect_colab, load_paths, mount_colab, prepare_outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths", help="Path configuration JSON; defaults to paths.json beside this file."
    )
    parser.add_argument("--stage", choices=("all", "rollouts", "train"), default="all")
    parser.add_argument(
        "--mount-drive", action="store_true", help="Mount Drive in an existing Colab environment."
    )
    parser.add_argument(
        "--disconnect", action="store_true", help="Disconnect the Colab runtime when finished."
    )
    return parser.parse_args()


def run_training(paths, stage="all"):
    # Check file locations before loading the model.
    if not Path(paths.dataset_csv).is_file():
        raise FileNotFoundError(
            f"Dataset not found: {paths.dataset_csv}. Set dataset_csv in the path configuration."
        )
    if stage == "train" and not Path(paths.trajectories_file).is_file():
        raise FileNotFoundError(
            f"Trajectories not found: {paths.trajectories_file}. Generate rollouts first."
        )
    if paths.export_copy_dir and Path(paths.final_model_dir).resolve() == Path(paths.export_copy_dir).resolve():
        raise ValueError("export_copy_dir must differ from final_model_dir, or be null.")
    prepare_outputs(paths)

    from interleaved_reasoning import data, model, ppo, prompts, rollouts

    # Load the model and dataset.
    base_model, tokenizer, policy_model, device = model.load_policy()
    prompts.model = base_model
    rollouts.device = device
    df, final_df = data.load_data(paths)

    # Collect rollouts, or load them for a training-only run.
    if stage in ("all", "rollouts"):
        trajectories = rollouts.collect_rollouts(policy_model, tokenizer, df, final_df, paths)
    else:
        with open(paths.trajectories_file, "rb") as handle:
            trajectories = pickle.load(handle)

    # Train and save the policy.
    if stage in ("all", "train"):
        if not trajectories:
            raise ValueError("No trajectories are available for training.")
        ppo.train_ppo(policy_model, tokenizer, df, trajectories, paths)
        ppo.save_final_model(policy_model, tokenizer, paths)
        if paths.export_copy_dir:
            shutil.copytree(paths.final_model_dir, paths.export_copy_dir, dirs_exist_ok=True)


def main():
    args = parse_args()
    paths = load_paths(args.paths)
    if args.mount_drive:
        mount_colab()
    run_training(paths, args.stage)
    if args.disconnect:
        disconnect_colab()


if __name__ == "__main__":
    main()
