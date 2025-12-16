from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

from gamebuilder.MB3_env import mariobros3_env
from utils.pretty_terminal.quiet_subproc_vec_env import QuietSubprocVecEnv
from utils.callbacks import (
    HyperparamSwitchOnLevelCallback,
    LevelGateEvalCallback,
    MaxHposPerEpisodeCallback,
    ResetStatsCallback,
    VideoOnXImproveCallback,
)
from utils.deaths.death_aggregate import (
    append_run_deaths_to_global,
)
from utils.deaths.plot_deaths_overlay_all import (
    render_deaths_overlay_all,
    render_deaths_overlay_from_jsonl,
)
from utils.run_dir import create_run_dir


def _safe_close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        # SubprocVecEnv workers can already be gone on Ctrl+C.
        return
    except Exception as e:
        print(f"[train] env.close() failed ({type(e).__name__}: {e})")


def generate_death_artifacts(*, run_dir: Path) -> None:
    """Generate per-run and global death overlays from a run directory."""

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
        try:
            out_all = render_deaths_overlay_from_jsonl(
                deaths_jsonl=Path("outputs") / "allDeath.jsonl",
                out=Path("outputs") / "all_deaths_overlay.png",
            )
            print(
                f"[train] Global deaths updated (+{added}); wrote: {out_all}"
            )
        except Exception as e:
            print(
                f"[train] Global deaths updated (+{added}); "
                f"skipped overlay ({type(e).__name__}: {e})"
            )
    except Exception as e:
        msg = (
            "[train] Skipped global deaths aggregation "
            f"({type(e).__name__}: {e})"
        )
        print(msg)


def train_ppo(*, profile: str = "laptop") -> None:
    max_steps = 10_000_000
    n_stack = 4
    custom_data_root = str(Path(__file__).resolve().parent / "retro_custom")

    if profile not in {"laptop", "pc"}:
        raise ValueError(
            f"Unknown profile: {profile!r} "
            "(expected 'laptop' or 'pc')"
        )

    if profile == "pc":
        n_envs = 10
        n_steps = 1024
        batch_size = 512
        n_epochs = 3
        learning_rate = 2e-4
        ent_coef = 0.002
        clip_range = 0.2
        gamma = 0.99
        gae_lambda = 0.95
        max_grad_norm = 0.5
        target_kl = 0.03
        device = "cuda"
    else:
        # Conservative defaults that match the previous training setup.
        n_envs = 4
        n_steps = 1024
        batch_size = 128
        n_epochs = 3
        learning_rate = 2e-4
        ent_coef = 0.002
        clip_range = 0.2
        gamma = 0.99
        gae_lambda = 0.95
        max_grad_norm = 0.5
        target_kl = 0.03
        device = "auto"

    run_dir = create_run_dir()
    print(f"[train] run_dir: {run_dir}")
    print(f"[train] profile: {profile}")
    print(f"[train] n_envs: {n_envs}")
    print(
        "[train] PPO: "
        f"n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, "
        f"lr={learning_rate}, ent={ent_coef}, clip={clip_range}, "
        f"gamma={gamma}, gae_lambda={gae_lambda}, "
        f"max_grad_norm={max_grad_norm}, device={device}"
    )
    print("[train] TensorBoard: tensorboard --logdir outputs/runs")

    stats_dir = Path(run_dir) / "stats"
    stats_csv = str(stats_dir / "episode_stats.csv")
    goal_cleared_jsonl = str(stats_dir / "goal_cleared.jsonl")
    videos_dir = str(run_dir / "videos")
    shared_switch_path = str(Path(run_dir) / "level_switch.json")

    venv = QuietSubprocVecEnv(
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
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        verbose=1,
        target_kl=target_kl,
        device=device,
        tensorboard_log=tb_logdir,
    )

    # Targeted exploration boost for the Level 1-2 pit bottleneck:
    # once training actually starts from Level 2 (or the Level2_Pit
    # checkpoint), increase entropy coefficient and lower LR to help discover
    # and retain the correct jump/box behavior.
    ent_coef_level2 = 0.02
    learning_rate_level2 = 1e-4

    try:
        try:
            model.learn(
                total_timesteps=max_steps,
                tb_log_name="ppo",
                callback=[
                    ResetStatsCallback(),
                    LevelGateEvalCallback(
                        custom_data_root=custom_data_root,
                        shared_switch_path=shared_switch_path,
                        required_successes=3,
                        eval_max_steps=6000,
                        deterministic=True,
                        cooldown_steps=50_000,
                        verbose=1,
                    ),
                    HyperparamSwitchOnLevelCallback(
                        trigger_episode_states=(
                            "1Player.World1.Level2",
                            "1Player.World1.Level2_Pit",
                            "1Player.World1.Level6",
                        ),
                        revert_episode_states=(
                            "1Player.World1.Level1",
                            "1Player.World1.Level3",
                        ),
                        ent_coef_after=ent_coef_level2,
                        learning_rate_after=learning_rate_level2,
                        verbose=1,
                    ),
                    MaxHposPerEpisodeCallback(
                        csv_path=stats_csv,
                        goal_cleared_jsonl_path=goal_cleared_jsonl,
                    ),
                    VideoOnXImproveCallback(
                        custom_data_root=custom_data_root,
                        out_dir=videos_dir,
                        n_stack=n_stack,
                        min_improvement_x=1,
                        min_episodes_before_trigger=0,
                        video_length_steps=1500,
                        fps=30,
                        deterministic=False,
                        verbose=1,
                    ),
                ],
            )
        except KeyboardInterrupt:
            print("[train] Aborted by user (Ctrl+C).")
            return

        model.save("ppo_super_mario_bros3")

        generate_death_artifacts(run_dir=Path(run_dir))

        # Short rollout for visual sanity-check.
        obs = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)  # type: ignore[arg-type]
            obs, _rewards, dones, _infos = venv.step(action)
            done = dones.any()
            venv.render()
    finally:
        _safe_close_env(venv)
