from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
)

from env.MB3_env import mariobros3_env

from utils.callbacks import MaxHposPerEpisodeCallback, ResetStatsCallback
from utils.runs import create_run_dir
from utils.plot_deaths_overlay_all import render_deaths_overlay_all
from utils.death_aggregate import (
    append_run_deaths_to_global,
    load_global_death_xs,
)
from utils.death_overlay import render_overlay


def train_ppo() -> None:
    max_steps = 10_000_000
    n_envs = 10
    n_stack = 4
    custom_data_root = str(Path(__file__).resolve().parent / "retro_custom")

    run_dir = create_run_dir()
    print(f"[train] run_dir: {run_dir}")
    print("[train] TensorBoard: tensorboard --logdir outputs/runs")

    stats_csv = str(run_dir / "stats" / "episode_stats.csv")

    venv = SubprocVecEnv(
        [
            lambda i=i, rd=str(run_dir): mariobros3_env(
                custom_data_root,
                rank=i,
                run_dir=rd,
            )
            for i in range(n_envs)
        ]
    )
    # Adds `info['episode']` with episode reward/length and standardizes
    # episode boundary bookkeeping for vectorized envs.
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=n_stack)

    tb_logdir = str(run_dir / "tb")

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        batch_size=256,
        device="cuda",
        ent_coef=0.01,
        n_epochs=3,
        n_steps=256,
        verbose=1,
        learning_rate=1e-4,
        clip_range=0.2,
        tensorboard_log=tb_logdir,
    )

    try:
        model.learn(
            total_timesteps=max_steps,
            tb_log_name="ppo",
            callback=[
                ResetStatsCallback(),
                MaxHposPerEpisodeCallback(csv_path=stats_csv),
            ],
        )
        model.save("ppo_super_mario_bros3")

        # Auto-generate deaths overlay for this run.
        try:
            out = render_deaths_overlay_all(
                deaths_dir=Path(run_dir) / "deaths",
                out=Path(run_dir) / "deaths_overlay.png",
            )
            print(f"[train] Wrote deaths overlay: {out}")
        except Exception as e:
            print(f"[train] Skipped deaths overlay ({type(e).__name__}: {e})")

        # Update global deaths file and regenerate a global overlay.
        try:
            added = append_run_deaths_to_global(run_dir=Path(run_dir))
            xs = load_global_death_xs()
            if xs.size > 0:
                out_all = render_overlay(
                    image_path=Path("assets/level_1-1.png"),
                    xs=xs,
                    ys=None,
                    out=Path("outputs") / "all_deaths_overlay.png",
                )
                print(
                    f"[train] Global deaths updated (+{added}); "
                    f"wrote: {out_all}"
                )
            else:
                print(f"[train] Global deaths updated (+{added}); no xs yet")
        except Exception as e:
            msg = (
                "[train] Skipped global deaths aggregation "
                f"({type(e).__name__}: {e})"
            )
            print(msg)

        # Short rollout for visual sanity-check.
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)  # type: ignore[arg-type]
            obs, _rewards, dones, _infos = venv.step(action)
            done = dones.any()
            venv.render()
    finally:
        venv.close()
