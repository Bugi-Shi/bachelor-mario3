from ppo import train_ppo
import argparse

'''
- Main.py -
- Entry point for training PPO on SuperMarioBros3-Nes environment.

python main.py --start # Start the training process
python main.py         # Also starts the training process
python main.py --load  # Load a previously saved model and continue training
'''


def main():
    parser = argparse.ArgumentParser(
        description="SuperMarioBros3-Nes PPO Training Script",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the training process",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load a previously saved model and continue training",
    )
    args = parser.parse_args()

    if args.start:
        print("Starte Mario PPO Training aus main.py ...")
        train_ppo()
        return
    elif args.load:
        print(
            "Lade ein zuvor gespeichertes Modell "
            "und setze das Training fort ..."
        )
        path = "ppo_super_mario_bros3"
        train_ppo(path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
