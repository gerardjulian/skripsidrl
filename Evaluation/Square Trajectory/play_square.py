# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with Square Trajectory, Telemetry Plotting, and Fall Detection."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL, Square Trajectory, and Fall Detection.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import math
import csv
from pathlib import Path
from datetime import datetime

# Matplotlib untuk plot
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# ============================================================================
# KELAS LOGGER DENGAN DETEKSI JATUH & MIN-MAX TICKS
# ============================================================================
class StepLogger:
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

    @staticmethod
    def _get_pitch_deg(quat_wxyz: torch.Tensor) -> float:
        try:
            from isaaclab.utils.math import euler_xyz_from_quat
            _, pitch, _ = euler_xyz_from_quat(quat_wxyz.unsqueeze(0))
            return float(torch.rad2deg(pitch[0]))
        except Exception:
            return float("nan")

    def record(self, env, time_s: float, fall_event: bool = False):
        base_env = env.unwrapped
        robot    = base_env.scene["robot"]

        lin_vel = robot.data.root_lin_vel_b[0]   
        ang_vel = robot.data.root_ang_vel_w[0]   
        pitch_d = self._get_pitch_deg(robot.data.root_quat_w[0])

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

    def save_csv(self) -> Path:
        path = self.log_dir / f"{self.timestamp}_telemetry.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def save_plot(self) -> "Path | None":
        if not self.rows:
            return None
            
        def comma_formatter(x, pos):
            return f"{x:g}".replace(".", ",")

        def apply_minmax_ticks(ax, base_ticks, val_min, val_max, color, fmt_str=".2f"):
            """Menggambar garis putus-putus dan anotasi nilai untuk Min dan Max aktual."""
            ax.set_yticks(sorted(set(base_ticks)))
            labels = [f"{v:g}".replace(".", ",") for v in sorted(set(base_ticks))]
            ax.set_yticklabels(labels, fontsize=7)
            y_bottom, y_top = ax.get_ylim()
            annotations = []
            
            # Garis & Teks Maksimum
            if val_max is not None and val_max <= y_top:
                annotations.append((val_max, 2))
                ax.axhline(val_max, color=color, lw=1.2, ls=":", alpha=0.9)
            
            # Garis & Teks Minimum
            if val_min is not None and val_min >= y_bottom:
                annotations.append((val_min, -8))
                ax.axhline(val_min, color=color, lw=1.2, ls=":", alpha=0.9)
                
            for val, va_offset in annotations:
                ax.annotate(format(val, fmt_str), xy=(0, val), xycoords=("axes fraction", "data"),
                            xytext=(4, va_offset), textcoords="offset points", fontsize=8, color=color,
                            fontweight="bold", ha="left", va="center")

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
        fig.suptitle(f"Square Trajectory | Durasi: {t[-1]:.1f}s | Jatuh: {len(fall_times)}x", fontsize=14, fontweight="bold")

        # --- KOTAK 1: Kecepatan Maju (Vel X) ---
        axes[0].plot(t, vx_a, label="Actual vel_x",  color="#2196F3", lw=1.5)
        axes[0].plot(t, vx_c, label="Command vel_x", color="#FF9800", lw=1.2, ls="--")
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].set_ylabel("Lin Vel X (m/s)")
        if vx_a:
            apply_minmax_ticks(axes[0], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], min(vx_a), max(vx_a), "#2196F3")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].grid(alpha=0.3)

        # --- KOTAK 2: Kecepatan Samping (Vel Y) ---
        axes[1].plot(t, vy_a, label="Actual vel_y",  color="#9C27B0", lw=1.5)
        axes[1].plot(t, vy_c, label="Command vel_y", color="#FF9800", lw=1.2, ls="--")
        axes[1].set_ylim(-1.0, 1.0)
        axes[1].set_ylabel("Lin Vel Y (m/s)")
        if vy_a:
            apply_minmax_ticks(axes[1], [-1.0, -0.5, 0.0, 0.5, 1.0], min(vy_a), max(vy_a), "#9C27B0")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].grid(alpha=0.3)

        # --- KOTAK 3: Kecepatan Putar (Yaw / Ang Vel Z) ---
        axes[2].plot(t, az_a, label="Actual ang_vel_z",  color="#00BCD4", lw=1.5)
        axes[2].plot(t, az_c, label="Command ang_vel_z", color="#FF9800", lw=1.2, ls="--")
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].set_ylabel("Ang Vel Z (rad/s)")
        if az_a:
            apply_minmax_ticks(axes[2], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], min(az_a), max(az_a), "#00BCD4")
        axes[2].legend(fontsize=8, loc="lower right")
        axes[2].grid(alpha=0.3)

        # --- KOTAK 4: Keseimbangan (Pitch) ---
        axes[3].plot(t, pitch, label="Pitch", color="#9C27B0", lw=1.5)
        axes[3].set_ylim(-30, 30)
        axes[3].set_ylabel("Pitch (deg)")
        axes[3].set_xlabel("Time (s)")
        if pitch:
            apply_minmax_ticks(axes[3], [-30, -20, -10, 0, 10, 20, 30], min(pitch), max(pitch), "#9C27B0", fmt_str=".1f")
        axes[3].legend(fontsize=8, loc="lower right")
        axes[3].grid(alpha=0.3)

        # Garis merah jatuh
        for ft in fall_times:
            for ax in axes:
                ax.axvline(ft, color="red", alpha=0.5, lw=1.0, ls="--")

        if fall_times:
            axes[0].annotate(
                f"⚠ Robot jatuh {len(fall_times)}x (garis vertikal merah)",
                xy=(0.02, 1.05), xycoords="axes fraction",
                fontsize=10, color="red",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )

        fmt = ticker.FuncFormatter(comma_formatter)
        axes[3].xaxis.set_major_formatter(fmt)
        plt.tight_layout()
        path = self.log_dir / f"{self.timestamp}_telemetry_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # =================================================================================
    # PERSIAPAN TRACKING LINTASAN & LOGGER
    # =================================================================================
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    
    total_time = 0.0
    timestep = 0
    fall_count = 0 
    
    durasi_maju = 4.0   
    durasi_belok = 2.0  
    durasi_fase = durasi_maju + durasi_belok
    
    x_history = []
    y_history = []
    
    base_env = env.unwrapped
    robot_asset = base_env.scene["robot"]
    command_term = base_env.command_manager.get_term("velocity")

    logger = StepLogger(log_dir=log_dir, duration=24.0)

    print(f"[INFO] Memulai simulasi bermanuver persegi selama 24 detik...")

    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            waktu_dalam_siklus = total_time % durasi_fase

            if waktu_dalam_siklus < durasi_maju:
                command_term.vel_command_b[:, 0] = 0.5 
                command_term.vel_command_b[:, 1] = 0.0
                command_term.vel_command_b[:, 2] = 0.0
            else:
                target_wz = (math.pi / 2.6) / durasi_belok 
                command_term.vel_command_b[:, 0] = 0.0  
                command_term.vel_command_b[:, 1] = 0.0
                command_term.vel_command_b[:, 2] = target_wz

            with torch.inference_mode():
                actions = policy(obs)
                step_returns = env.step(actions)
                obs = step_returns[0]

            # -------------------------------------------------------------
            # DETEKSI JATUH
            # -------------------------------------------------------------
            if len(step_returns) == 5: 
                dones, infos = step_returns[3], step_returns[4]
            elif len(step_returns) == 4: 
                dones, infos = step_returns[2], step_returns[3]
            else:
                dones, infos = None, {}

            is_done = False
            timeout = False
            
            if dones is not None and hasattr(dones, "any"):
                is_done = bool(dones.any())
            if "time_outs" in infos and hasattr(infos["time_outs"], "any"):
                timeout = bool(infos["time_outs"].any())

            fall_event = False
            if is_done and not timeout:
                fall_event = True
                fall_count += 1
                print(f"[Logger] ⚠ Robot jatuh ke-{fall_count} pada detik {total_time:.2f}s → Posisi di-reset otomatis.")
                obs = env.get_observations()

            # REKAM DATA
            logger.record(env, time_s=total_time, fall_event=fall_event)

            root_pos_w = robot_asset.data.root_pos_w[0]
            x_history.append(float(root_pos_w[0].item()))
            y_history.append(float(root_pos_w[1].item()))
            
            total_time += dt
            timestep += 1
            
            if args_cli.video and timestep == args_cli.video_length:
                break
                
            if total_time >= 24.0:
                print("\n[INFO] 1 Lintasan persegi selesai. Menghentikan simulasi...")
                break   

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n[INFO] Simulasi dihentikan secara manual (Ctrl+C).")
    finally:
        env.close()

        print(f"\n[Logger] ══════════════════════════════════════")
        print(f"[Logger] Total waktu berjalan  : {total_time:.2f}s")
        print(f"[Logger] Total kejadian jatuh  : {fall_count}x")
        print(f"[Logger] ══════════════════════════════════════")

   # =================================================================================
        # OUTPUT 1: PLOT LINTASAN (X-Y)
        # =================================================================================
        if len(x_history) > 0:
            print("\n[INFO] Menyimpan grafik...")
            
            fig_xy = plt.figure(figsize=(8, 8))
            plt.plot(x_history, y_history, label="Jejak Robot", color="#2196F3", linewidth=2)
            plt.scatter([x_history[0]], [y_history[0]], color="green", s=100, zorder=5, label="Start")
            plt.scatter([x_history[-1]], [y_history[-1]], color="red", s=100, zorder=5, label="End")
            
            plt.title(f"Lintasan Robot (XY Plane)\nDurasi: {total_time:.1f} detik | Jatuh: {fall_count}x", fontsize=14, fontweight="bold")
            plt.xlabel("Posisi X (meter)", fontsize=12)
            plt.ylabel("Posisi Y (meter)", fontsize=12)
            plt.axis('equal')
            plt.grid(True, alpha=0.5, linestyle="--")
            plt.legend(loc="best")
            
            plot_xy_path = os.path.join(log_dir, f"{logger.timestamp}_path_square.png")
            plt.savefig(plot_xy_path, dpi=200, bbox_inches="tight")
            plt.close(fig_xy)
            print(f"[SUCCESS] Grafik lintasan: {plot_xy_path}")

        # =================================================================================
        # OUTPUT 2: PLOT TELEMETRI & CSV
        # =================================================================================
        if logger.rows:
            csv_path = logger.save_csv()
            plot_tel_path = logger.save_plot()
            print(f"[SUCCESS] CSV Telemetri  : {csv_path}")
            if plot_tel_path:
                print(f"[SUCCESS] Plot Telemetri : {plot_tel_path}\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
