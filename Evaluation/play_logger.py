# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
play_logger.py — Drop-in pengganti play.py dengan logging otomatis
==================================================================
Merekam selama 30 detik:
  - Kecepatan linear X dan Y (aktual vs command)
  - Sudut Pitch (condong maju/mundur, rotasi sumbu Y)

Jika robot jatuh sebelum 30 detik → reset otomatis dan lanjut
hingga total waktu 30 detik terpenuhi.

CARA PAKAI:
    python play_logger.py --task=Centaur-Balancing-v10-Play --num_envs=1

OUTPUT (di folder play_logs/):
    YYYYMMDD_HHMMSS.csv          ← data mentah per-step
    YYYYMMDD_HHMMSS_plot.png     ← grafik otomatis
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── Argparse ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play + Log RSL-RL agent (Centaur).")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,   default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs",       type=int,   default=None)
parser.add_argument("--task",           type=str,   default=None)
parser.add_argument("--agent",          type=str,   default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",           type=int,   default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time",      action="store_true", default=False)
parser.add_argument("--log_dir",        type=str,   default="play_logs")
parser.add_argument("--no_plot",        action="store_true")

# Durasi total recording (detik) — default 30s sesuai kebutuhan
parser.add_argument("--duration",       type=float, default=30.0,
                    help="Total durasi recording dalam detik (default: 30)")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Import setelah sim aktif ───────────────────────────────────────────────
import csv
import time
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class StepLogger:
    """
    Catat 5 kolom per-step selama total DURATION detik:
        time_s      — waktu kumulatif sejak mulai (detik)
        cmd_vel_x   — kecepatan linear X yang diperintahkan (m/s)
        cmd_vel_y   — kecepatan linear Y yang diperintahkan (m/s)
        act_vel_x   — kecepatan linear X aktual robot (m/s)
        act_vel_y   — kecepatan linear Y aktual robot (m/s)
        pitch_deg   — sudut pitch aktual robot (derajat, + = condong maju)
        fall_event  — 1 jika step ini adalah saat robot jatuh, 0 lainnya
    """

    FIELDS = [
        "time_s",
        "cmd_vel_x", "cmd_vel_y",
        "act_vel_x", "act_vel_y",
        "cmd_ang_vel_z", "act_ang_vel_z",
        "pitch_deg",
        "fall_event",
    ]

    def __init__(self, log_dir: str, duration: float):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.rows: list[dict] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── konversi quat → pitch (derajat) ───────────────────────────────────
    @staticmethod
    def _get_pitch_deg(quat_wxyz: torch.Tensor) -> float:
        """Ekstrak sudut pitch dari quaternion (rotasi sumbu Y)."""
        try:
            from isaaclab.utils.math import euler_xyz_from_quat
            _, pitch, _ = euler_xyz_from_quat(quat_wxyz.unsqueeze(0))
            return float(torch.rad2deg(pitch[0]))
        except Exception:
            return float("nan")

    # ── record satu step ──────────────────────────────────────────────────
    def record(self, env, time_s: float, fall_event: bool = False):
        base_env = env.unwrapped
        robot    = base_env.scene["robot"]

        # Kecepatan aktual (body frame)
        lin_vel = robot.data.root_lin_vel_b[0]   # (3,) — x, y, z
        ang_vel = robot.data.root_ang_vel_w[0]   # (3,) — roll, pitch, yaw

        # Sudut pitch
        pitch_d = self._get_pitch_deg(robot.data.root_quat_w[0])

        # Command velocity
        try:
            cmd = base_env.command_manager.get_command("velocity")[0]
            cx, cy = float(cmd[0]), float(cmd[1])
            cz = float(cmd[2]) if len(cmd) > 2 else 0.0
        except Exception:
            cx = cy = cz = float("nan")

        self.rows.append({
            "time_s":        round(time_s, 4),
            "cmd_vel_x":     round(cx, 4),
            "cmd_vel_y":     round(cy, 4),
            "act_vel_x":     round(float(lin_vel[0]), 4),
            "act_vel_y":     round(float(lin_vel[1]), 4),
            "cmd_ang_vel_z": round(cz, 4),
            "act_ang_vel_z": round(float(ang_vel[2]), 4),
            "pitch_deg":     round(pitch_d, 3),
            "fall_event":    1 if fall_event else 0,
        })

    # ── simpan CSV ────────────────────────────────────────────────────────
    def save_csv(self) -> Path:
        path = self.log_dir / f"{self.timestamp}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\n[Logger] ✓ CSV  → {path}  ({len(self.rows)} baris)")
        return path

    # ── simpan plot PNG ───────────────────────────────────────────────────
    def save_plot(self) -> "Path | None":
        if not self.rows:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            print("[Logger] matplotlib tidak tersedia, skip plot.")
            return None

        # Formatter: ganti titik desimal dengan koma (misal 0.3 → 0,3)
        def comma_formatter(x, pos):
            return f"{x:g}".replace(".", ",")

        # Helper: pasang tick bawaan (tanpa min/max), lalu anotasi min/max
        # di dalam plot area — menempel di tepi kiri, sebelum data dimulai.
        # Helper: pasang tick bawaan, lalu anotasi min/max dengan cerdas
        def apply_minmax_ticks(ax, base_ticks, val_min, val_max, color, fmt_str=".2f"):
            """
            Smart Min/Max: Hanya menggambar garis dan teks anotasi 
            jika nilainya masih berada di dalam batas (limit) sumbu Y kotak grafik.
            """
            ax.set_yticks(sorted(set(base_ticks)))
            labels = [f"{v:g}".replace(".", ",") for v in sorted(set(base_ticks))]
            ax.set_yticklabels(labels, fontsize=7)

            # Ambil batas atap dan lantai kotak grafik saat ini
            y_bottom, y_top = ax.get_ylim()

            annotations = []
            
            # Cek & Gambar MAX (hanya jika nilainya tidak melebihi atap grafik)
            if val_max is not None and val_max <= y_top:
                annotations.append((val_max, 2))
                ax.axhline(val_max, color=color, lw=1.0, ls=":", alpha=0.8)
                
            # Cek & Gambar MIN (hanya jika nilainya tidak tembus lantai grafik)
            if val_min is not None and val_min >= y_bottom:
                annotations.append((val_min, -8))
                ax.axhline(val_min, color=color, lw=1.0, ls=":", alpha=0.8)

            # Tuliskan teks anotasinya
            for val, va_offset in annotations:
                ax.annotate(
                    format(val, fmt_str),
                    xy=(0, val),                    
                    xycoords=("axes fraction", "data"),
                    xytext=(4, va_offset),           
                    textcoords="offset points",
                    fontsize=7, color=color,
                    fontweight="bold", ha="left", va="center"
                )

        t         = [r["time_s"]        for r in self.rows]
        vx_a      = [r["act_vel_x"]     for r in self.rows]
        vx_c      = [r["cmd_vel_x"]     for r in self.rows]
        vy_a      = [r["act_vel_y"]     for r in self.rows]
        vy_c      = [r["cmd_vel_y"]     for r in self.rows]
        az_a      = [r["act_ang_vel_z"] for r in self.rows]
        az_c      = [r["cmd_ang_vel_z"] for r in self.rows]
        pitch     = [r["pitch_deg"]     for r in self.rows]

        fall_times = [r["time_s"] for r in self.rows if r["fall_event"] == 1]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"Play Log  —  Bergerak Maju 1,0 m/s\n"
            f"Arm Swing 90° | Durasi: 30s | Jatuh: {len(fall_times)}x",
            fontsize=12, fontweight="bold"
        )

        # ── Panel 1: Kecepatan Linear X ───────────────────────────────────
        axes[0].plot(t, vx_a, label="Actual vel_x",  color="#2196F3", lw=1.5)
        axes[0].plot(t, vx_c, label="Command vel_x", color="#FF9800", lw=1.2, ls="--")
        axes[0].set_ylim(-2, 2)
        conv_start = int(len(vx_a) * 0.05)
        vx_conv = vx_a[conv_start:]
        vx_cmd_val = float(vx_c[0]) if vx_c else 0.0
        if vx_conv:
            vx_max_conv = max(vx_conv)
            vx_min_conv = min(vx_conv)
            base_0 = sorted(set([-2, -1, 0, 1, 2] + [round(vx_cmd_val, 2)]))
            # Panggil fungsi pintar (garis putus-putus sudah otomatis digambar di dalam fungsi)
            apply_minmax_ticks(axes[0], base_0, vx_min_conv, vx_max_conv, "#2196F3")
            for lbl, val in zip(axes[0].get_yticklabels(), axes[0].get_yticks()):
                if abs(val - vx_cmd_val) < 0.01:
                    lbl.set_color("#FF9800")
                    lbl.set_fontweight("bold")
        else:
            axes[0].set_yticks([-2, -1, 0, 1, 2])
        axes[0].set_ylabel("Lin Vel X (m/s)")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].grid(alpha=0.3)

        # ── Panel 2: Kecepatan Linear Y ───────────────────────────────────
        axes[1].plot(t, vy_a, label="Actual vel_y",  color="#9C27B0", lw=1.5)
        axes[1].plot(t, vy_c, label="Command vel_y", color="#FF9800", lw=1.2, ls="--")
        axes[1].set_ylim(-2, 2)
        vy_cmd_val = float(vy_c[0]) if vy_c else 0.0
        if vy_a:
            vy_max = max(vy_a)
            vy_min = min(vy_a)
            base_1 = sorted(set([-2, -1, 0, 1, 2] + [round(vy_cmd_val, 2)]))
            apply_minmax_ticks(axes[1], base_1, vy_min, vy_max, "#9C27B0")
            for lbl, val in zip(axes[1].get_yticklabels(), axes[1].get_yticks()):
                if abs(val - vy_cmd_val) < 0.01:
                    lbl.set_color("#FF9800")
                    lbl.set_fontweight("bold")
        else:
            axes[1].set_yticks([-2, -1, 0, 1, 2])
        axes[1].set_ylabel("Lin Vel Y (m/s)")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].grid(alpha=0.3)

        # ── Panel 3: Angular Velocity Z ───────────────────────────────────
        axes[2].plot(t, az_a, label="Actual ang_vel_z",  color="#00BCD4", lw=1.5)
        axes[2].plot(t, az_c, label="Command ang_vel_z", color="#FF9800", lw=1.2, ls="--")
        axes[2].axhline(0, color="gray", lw=0.8, ls=":")
        axes[2].set_ylim(-2, 2)
        az_cmd_val = float(az_c[0]) if az_c else 0.0
        if az_a:
            az_max = max(az_a)
            az_min = min(az_a)
            base_2 = sorted(set([-2, -1, 0, 1, 2] + [round(az_cmd_val, 2)]))
            apply_minmax_ticks(axes[2], base_2, az_min, az_max, "#00BCD4")
            for lbl, val in zip(axes[2].get_yticklabels(), axes[2].get_yticks()):
                if abs(val - az_cmd_val) < 0.01:
                    lbl.set_color("#FF9800")
                    lbl.set_fontweight("bold")
        else:
            axes[2].set_yticks([-2, -1, 0, 1, 2])
        axes[2].set_ylabel("Ang Vel Z (rad/s)")
        axes[2].legend(fontsize=8, loc="lower right")
        axes[2].grid(alpha=0.3)

        # ── Panel 4: Pitch ────────────────────────────────────────────────
        axes[3].plot(t, pitch, label="Pitch", color="#9C27B0", lw=1.5)
        axes[3].axhline(0,  color="gray",   lw=0.8, ls=":")
        axes[3].set_ylim(-30, 30)
        if pitch:
            p_max = max(pitch)
            p_min = min(pitch)
            apply_minmax_ticks(axes[3], [-30, -20, -10, 0, 10, 20, 30],
                               p_min, p_max, "#9C27B0", fmt_str=".1f")
        else:
            axes[3].set_yticks([-30, -20, -10, 0, 10, 20, 30])
        axes[3].set_ylabel("Pitch (deg)")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(fontsize=8, loc="lower right")
        axes[3].grid(alpha=0.3)

        # Garis merah vertikal di setiap kejadian jatuh
        for ft in fall_times:
            for ax in axes:
                ax.axvline(ft, color="red", alpha=0.5, lw=1.0, ls="--")
                
     
        # ==========================================================
        # Terapkan formatter koma hanya ke sumbu X
        fmt = ticker.FuncFormatter(comma_formatter)
        axes[3].xaxis.set_major_formatter(fmt)

        # Anotasi jumlah jatuh — hanya muncul jika robot benar-benar jatuh
        if fall_times:
            axes[0].annotate(
                f"⚠ Robot jatuh {len(fall_times)}x (garis merah putus-putus)",
                xy=(0.35, 1.05), xycoords="axes fraction",
                fontsize=10, color="red",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )

        plt.tight_layout()
        path = self.log_dir / f"{self.timestamp}_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Logger] ✓ Plot → {path}")
        return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: "ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg",
         agent_cfg: RslRlBaseRunnerCfg):
    """Play + log 30 detik dengan auto-reset saat robot jatuh."""

    # ── Setup identik play.py ─────────────────────────────────────────────
    task_name       = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (args_cli.num_envs
                               if args_cli.num_envs is not None
                               else env_cfg.scene.num_envs)
    env_cfg.seed       = agent_cfg.seed
    env_cfg.sim.device = (args_cli.device
                          if args_cli.device is not None
                          else env_cfg.sim.device)

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint tidak tersedia.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir         = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder":   os.path.join(log_dir, "videos", "play"),
            "step_trigger":   lambda step: step == 0,
            "video_length":   args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                                log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(),
                                    log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer,
                         path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer,
                          path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ── Setup logger & timer ──────────────────────────────────────────────
    logger     = StepLogger(args_cli.log_dir, duration=args_cli.duration)
    total_time = 0.0          # akumulator waktu kumulatif (detik)
    fall_count = 0

    print(f"\n[Logger] Durasi recording : {args_cli.duration}s")
    print(f"[Logger] dt per step      : {dt:.4f}s")
    print(f"[Logger] Estimasi step    : ~{int(args_cli.duration / dt)} step")
    print(f"[Logger] Output           : {Path(args_cli.log_dir).resolve()}")
    print(f"[Logger] Jika robot jatuh → reset otomatis, waktu tetap berjalan")
    print(f"[Logger] Tekan Ctrl+C untuk stop lebih awal\n")

    # ── Simulation loop ───────────────────────────────────────────────────
    obs = env.get_observations()
    try:
        while simulation_app.is_running() and total_time < args_cli.duration:
            start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)

                # Catat data state saat ini
                logger.record(env, time_s=total_time, fall_event=False)

                # Tangkap seluruh output step secara dinamis tanpa unpack statis
                step_returns = env.step(actions)
                obs = step_returns[0]

            # Tambah waktu kumulatif
            total_time += dt

            # ── Cek apakah robot jatuh dengan aman ────────────────────────
            is_done = False
            timeout = False
            
            # Deteksi arsitektur output dari wrapper RSL-RL
            if len(step_returns) == 5: 
                # (obs, priv_obs, rewards, dones, infos)
                dones, infos = step_returns[3], step_returns[4]
            elif len(step_returns) == 4: 
                # (obs, rewards, dones, infos)
                dones, infos = step_returns[2], step_returns[3]
            else:
                dones, infos = None, {}

            # Konversi tensor done
            if dones is not None and hasattr(dones, "any"):
                is_done = bool(dones.any())
            
            # Ekstrak info timeout (jika tersedia)
            if "time_outs" in infos and hasattr(infos["time_outs"], "any"):
                timeout = bool(infos["time_outs"].any())

            fallen = False
            done = is_done
            
            if done:
                # Robot dinyatakan jatuh jika Done terjadi BUKAN karena Timeout
                if not timeout:
                    fallen = True

                if fallen:
                    fall_count += 1
                    # Tandai baris logger terakhir bahwa terjadi kejadian jatuh
                    logger.rows[-1]["fall_event"] = 1
                    sisa = args_cli.duration - total_time
                    print(f"[Logger] ⚠ Robot jatuh ke-{fall_count}  "
                          f"| t={total_time:.2f}s  | sisa={sisa:.2f}s  → reset...")
                else:
                    sisa = args_cli.duration - total_time
                    print(f"[Logger] ↩ Episode selesai (bukan jatuh)  "
                          f"| t={total_time:.2f}s  | sisa={sisa:.2f}s  → reset...")
                
                # Update observation setelah reset otomatis
                obs = env.get_observations()

            # Progress setiap 5 detik
            if len(logger.rows) % int(5.0 / dt) == 0:
                pct = min(100, total_time / args_cli.duration * 100)
                print(f"[Logger] Progress: {total_time:.1f}s / "
                      f"{args_cli.duration}s  ({pct:.0f}%)")

            # Real-time delay
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Logger] Dihentikan manual (Ctrl+C)")
    finally:
        env.close()

        print(f"\n[Logger] ══════════════════════════════════════")
        print(f"[Logger] Total waktu terekam : {total_time:.2f}s")
        print(f"[Logger] Total step          : {len(logger.rows)}")
        print(f"[Logger] Robot jatuh         : {fall_count}x")
        print(f"[Logger] ══════════════════════════════════════")

        if logger.rows:
            logger.save_csv()
            if not args_cli.no_plot:
                logger.save_plot()
            print(f"[Logger] Semua file ada di: {logger.log_dir.resolve()}\n")
        else:
            print("[Logger] Tidak ada data yang terekam.")


if __name__ == "__main__":
    main()
    simulation_app.close()
