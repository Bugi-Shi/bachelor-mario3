import argparse
from pathlib import Path

from ppo import generate_death_artifacts, train_ppo
from utils.runs import latest_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mario PPO training entrypoint")
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

    print("Starte Mario PPO Training aus main.py ...")
    train_ppo()
    return


if __name__ == "__main__":
    main()
