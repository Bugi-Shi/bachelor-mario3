import argparse
from pathlib import Path

from sandbox import generate_death_artifacts, train_ppo
from utils.run_dir import latest_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mario PPO training entrypoint"
    )

    hw = parser.add_mutually_exclusive_group()
    hw.add_argument(
        "--laptop",
        dest="profile",
        action="store_const",
        const="laptop",
        help="Use conservative training hyperparameters for laptops.",
    )
    hw.add_argument(
        "--pc",
        dest="profile",
        action="store_const",
        const="pc",
        help="Use faster training hyperparameters for PCs (more envs/batch).",
    )

    parser.set_defaults(profile="pc")

    parser.add_argument(
        "--death",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Generate death overlays from an existing run. "
            "Use '--death' for the latest run or '--death <run_dir>' "
            "for a specific outputs/runs/<...> folder."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.death is not None:
        if args.death == "latest":
            run_dir = latest_run_dir()
            if run_dir is None:
                raise SystemExit(
                    "No run directory found under outputs/runs. "
                    "Run training first or pass --death <run_dir>."
                )
        else:
            run_dir = Path(str(args.death)).expanduser().resolve()
            if not run_dir.exists() or not run_dir.is_dir():
                raise SystemExit(f"Run dir not found: {run_dir}")

        print(f"[death] Generating overlays for: {run_dir}")
        generate_death_artifacts(run_dir=run_dir)
        return

    profile = str(args.profile)
    print(f"Starte Mario PPO Training aus main.py (profile={profile}) ...")
    train_ppo(profile=profile)
    return


if __name__ == "__main__":
    main()
