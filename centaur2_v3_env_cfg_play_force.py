"""
Play/Test configuration for Centaur robot.
This config removes the exclusive command logic so you can manually control velocities.

Usage:
    python play.py --task=Centaur-Balancing-v10-Play --num_envs=16
"""
import torch
from .centaur2_v3_env_cfg import (
    CentaurBalancingEnvCfg,
    CentaurSceneCfg,
    ActionsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import mdp
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab.managers import SceneEntityCfg

def apply_discrete_force_x(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, base_force: float = 20.0, force_step: float = 20.0, duration: float = 0.1, period: float = 3.0):
    """Fungsi kustom untuk gaya bergantian dengan metronom waktu global."""
    asset = env.scene[asset_cfg.name]
    
    # 1. Inisialisasi variabel memori
    if not hasattr(env, "_force_global_time"):
        env._force_global_time = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        # PERBAIKAN: Gunakan zeros_like agar otomatis bertipe Long (integer)
        env._prev_eps_len = torch.zeros_like(env.episode_length_buf)
        
        env._force_sign_toggle = torch.ones((env.num_envs, 1), device=env.device) * -1
        env._last_fire_time = torch.zeros((env.num_envs,), device=env.device) - period
        env._force_multiplier = torch.zeros((env.num_envs, 1), device=env.device)
        
    # --- METRONOM WAKTU GLOBAL ---
    # delta_steps sekarang dijamin bertipe sama dengan episode_length_buf
    delta_steps = env.episode_length_buf - env._prev_eps_len
    
    is_reset = delta_steps < 0
    # PERBAIKAN: Konversi sumber ke tipe data destinasi secara eksplisit
    delta_steps[is_reset] = env.episode_length_buf[is_reset].to(delta_steps.dtype)
    
    env._force_global_time += delta_steps * env.step_dt
    env._prev_eps_len = env.episode_length_buf.clone()
    
    current_time = env._force_global_time
    # ----------------------------------------------
    
    forces = torch.zeros((env.num_envs, asset.num_bodies, 3), device=env.device)
    
    time_since_last = current_time - env._last_fire_time
    trigger_now = time_since_last >= period
    
    if trigger_now.any():
        env._force_sign_toggle[trigger_now] *= -1
        env._last_fire_time[trigger_now] += period 
        env._force_multiplier[trigger_now] += 1.0
        
        if trigger_now[0]:
             arah = "DEPAN (+X)" if env._force_sign_toggle[0].item() > 0 else "BELAKANG (-X)"
             current_mag = base_force + (env._force_multiplier[0].item() - 1.0) * force_step
             print(f"[External Force] Waktu: {current_time[0].item():.1f}s | {current_mag} N ke {arah} selama {duration}s!")

    time_since_last_updated = current_time - env._last_fire_time
    is_active = (time_since_last_updated >= 0.0) & (time_since_last_updated <= duration)
    
    active_envs = is_active.nonzero(as_tuple=True)[0]
    
    if len(active_envs) > 0:
        signs = env._force_sign_toggle[active_envs]
        mults = env._force_multiplier[active_envs]
        current_mags = base_force + (mults - 1.0) * force_step
        forces[active_envs, asset_cfg.body_ids[0], 0] = (current_mags * signs).squeeze()

    asset.set_external_force_and_torque(
        forces=forces,
        torques=torch.zeros_like(forces),
        env_ids=torch.arange(env.num_envs, device=env.device)
    )
    
# EVENT CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class PlayEventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    push_robot_force = EventTerm(
        func=apply_discrete_force_x,
        mode="interval",
        interval_range_s=(0.01, 0.01), 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "base_force": 40.0, # Nilai gaya pada tembakan pertama
            "force_step": 40.0, # Nilai penambahan gaya untuk tembakan berikutnya
            "duration": 0.1,    # Durasi gaya sesaat
            "period": 3.0       # Ditembakkan setiap 3 detik
        },
    )
# ============================================================================
# COMMAND CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass  
class PlayCommandsCfg:
    """
    Commands for play mode.
    Set the exact velocity you want to test here.
    """
    
    velocity = vel_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,    # No random standing in play mode
        rel_heading_envs=0.0,
        ranges=vel_mdp.UniformVelocityCommandCfg.Ranges(
            # ============================================================
            # CHANGE THESE VALUES TO TEST DIFFERENT VELOCITIES:
            # ============================================================
            lin_vel_x=(0.0, 0.0),    # Forward / Backward
            lin_vel_y=(0.0, 0.0),    # No lateral (can't strafe)
            ang_vel_z=(0.0, 0.0),    # Turning
            heading=(0.0, 0.0),
            # ============================================================
  
        ),
        resampling_time_range=(100.0, 100.0), 
        debug_vis=True  
    )


# ============================================================================
# MAIN PLAY ENVIRONMENT CONFIGURATION
# ============================================================================
@configclass
class CentaurPlayEnvForceCfg(CentaurBalancingEnvCfg):
    """
    Play configuration for Centaur - inherits from training config
    but removes exclusive command logic and uses fewer environments.
    """
    
    # Use play-specific configs
    events: PlayEventCfg = PlayEventCfg()
    commands: PlayCommandsCfg = PlayCommandsCfg()
    
    def __post_init__(self) -> None:
        """Post initialization for play mode."""
        # Call parent post_init first
        super().__post_init__()
        
        # Override with play-specific settings
        self.scene.env_spacing = 4.0
        
        # Better camera angle for watching
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        
        # Longer episodes for manual testing
        self.episode_length_s = 31  # 30 seconds instead of 12
