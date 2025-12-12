import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SuperMarioBros3 PPO: train or plot death overlay",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="Run PPO training (calls train_ppo())",
    )
    group.add_argument(
        "--death",
        action="store_true",
        help="Generate deaths overlay image from outputs/deaths/*.jsonl",
    )
    args = parser.parse_args()

    if args.train:
        from ppo import train_ppo

        print("Starte Mario PPO Training aus main.py ...")
        train_ppo()
        return

    if args.death:
        from utils.plot_deaths_overlay_all import render_deaths_overlay_all

        try:
            out = render_deaths_overlay_all()
        except FileNotFoundError as e:
            print(e)
            print(
                "No deaths logs found yet. Run `python main.py --train` "
                "to generate outputs/deaths/*.jsonl first."
            )
            return
        print(f"Wrote: {out}")
        return


if __name__ == "__main__":
    main()
