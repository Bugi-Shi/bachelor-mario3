import retro
import time

def test_jump(steps=200, sleep=0.02):
    """
    Lässt Mario im SuperMarioBros3-Nes-v0 Environment springen.
    steps: Wie viele Schritte ausgeführt werden sollen.
    sleep: Wartezeit zwischen Frames (für Sichtbarkeit).
    """
    
    env = retro.make("SuperMarioBros3-Nes-v0")
    obs, info = env.reset()

    print("Mario Jump Test gestartet ...")

    # NES Button Layout:
    # ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    # Wir wollen: A = Sprung
    action = [0] * 9
    action[8] = 1  # A drücken

    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step}, Reward: {reward}")

        if terminated or truncated:
            print("Level neu gestartet.")
            obs, info = env.reset()

        time.sleep(sleep)

    env.close()
    print("Jump Test erfolgreich abgeschlossen!")
