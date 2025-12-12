import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


def load_episode_stats(jsonl_path: str):
    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_reward_curve(episodes, outdir):
    rewards = [e["total_reward"] for e in episodes]
    plt.figure()
    plt.plot(rewards)
    ma_window = min(100, max(5, len(rewards) // 10))
    ma = moving_average(rewards, window=ma_window)
    if len(ma) > 1:
        start = len(rewards) - len(ma) + 1
        end = len(rewards) + 1
        plt.plot(range(start, end), ma)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_curve.png"), dpi=200)
    plt.close()


def plot_max_hpos(episodes, outdir):
    hpos = [
        e["max_hpos"] if e["max_hpos"] is not None else np.nan
        for e in episodes
    ]
    plt.figure()
    plt.plot(hpos)
    plt.title("Max hpos per Episode")
    plt.xlabel("Episode")
    plt.ylabel("max(hpos)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "max_hpos.png"), dpi=200)
    plt.close()


def plot_action_histogram(episodes, outdir, top_k=15):
    # aggregate counts over all episodes
    agg = {}
    for e in episodes:
        for k, v in e.get("action_counts", {}).items():
            agg[k] = agg.get(k, 0) + int(v)

    if not agg:
        return

    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(f"Top-{top_k} Actions (aggregated)")
    plt.xlabel("Action encoding")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "action_hist.png"), dpi=200)
    plt.close()


def plot_death_scatter(death_positions, outdir):
    # death_positions: (hpos, vpos, episode, step)
    if death_positions.size == 0:
        return
    hpos = death_positions[:, 0]
    vpos = death_positions[:, 1]

    plt.figure()
    plt.scatter(hpos, vpos, marker=MarkerStyle("x"), alpha=0.25)
    plt.title("Death positions (scatter)")
    plt.xlabel("hpos")
    plt.ylabel("vpos")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "death_scatter.png"), dpi=200)
    plt.close()


def plot_death_heatmap(death_positions, outdir, bin_hpos=32, bin_vpos=16):
    """
    Simple heatmap:
      - bin_hpos: pixel width per bin on x-axis (hpos)
      - bin_vpos: pixel height per bin on y-axis (vpos)

    If you don't have vpos, it will be 0; heatmap becomes a line (still okay).
    """
    if death_positions.size == 0:
        return

    hpos = death_positions[:, 0]
    vpos = death_positions[:, 1]

    # filter invalid
    mask = np.isfinite(hpos) & np.isfinite(vpos) & (hpos >= 0)
    hpos = hpos[mask]
    vpos = vpos[mask]
    if len(hpos) == 0:
        return

    xbins = max(10, int((hpos.max() - hpos.min()) / bin_hpos) + 1)
    ybins = max(10, int((vpos.max() - vpos.min()) / bin_vpos) + 1)

    H, xedges, yedges = np.histogram2d(hpos, vpos, bins=[xbins, ybins])

    plt.figure(figsize=(10, 4))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.title("Death heatmap")
    plt.xlabel("hpos")
    plt.ylabel("vpos")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "death_heatmap.png"), dpi=200)
    plt.close()


def main(log_dir="logs/run_01", outdir="plots/run_01"):
    os.makedirs(outdir, exist_ok=True)

    jsonl_path = os.path.join(log_dir, "episode_stats.jsonl")
    death_path = os.path.join(log_dir, "death_positions.npy")

    episodes = load_episode_stats(jsonl_path)
    death_positions = (
        np.load(death_path)
        if os.path.exists(death_path)
        else np.zeros((0, 4), dtype=np.float32)
    )

    plot_reward_curve(episodes, outdir)
    plot_max_hpos(episodes, outdir)
    plot_action_histogram(episodes, outdir)
    plot_death_scatter(death_positions, outdir)
    plot_death_heatmap(death_positions, outdir)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
