from ppo import train_ppo


# Ensure we run inside the project venv (if available). This will re-exec the
# current process using `project/bin/python` when needed, so later imports
# (like `retro`) resolve correctly.

# - Main.py -
#
# # NES Button Layout:
# ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
#
# Advance running:
# action[0] = 1  # B drücken
# action[7] = 1  # RIGHT drücken
#
# action[8] = 1  # A drücken
#
# Ducking:
# action[5] = 1  # DOWN drücken


def main():
    print("Starte Mario PPO Training aus main.py ...")
    train_ppo()


if __name__ == "__main__":
    main()
