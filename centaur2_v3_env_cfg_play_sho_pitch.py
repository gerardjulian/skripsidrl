"""
Play/Test configuration for Centaur robot.
This config removes the exclusive command logic so you can manually control velocities,
and injects a custom robustness test for the shoulder joints.

Usage:
    python play.py --task=Centaur-Balancing-v10-Play --num_envs=16
"""

import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import mdp
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from .centaur2_v3_env_cfg import (
    CentaurBalancingEnvCfg,
    CentaurSceneCfg,
    ActionsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)

# ============================================================================
# CUSTOM EVENT UNTUK DISTURBANCE BAHU (ROBUSTNESS TEST)
# ============================================================================
def swing_shoulders_event(env, env_ids, asset_cfg: SceneEntityCfg, amplitude_deg: float, freq_hz: float):
    """
    Fungsi custom untuk menggerakkan bahu secara sinusoidal dari -amplitude hingga +amplitude.
    Berfungsi memindahkan Center of Mass secara dinamis untuk menguji keseimbangan.
    """
    asset = env.scene[asset_cfg.name]
    
    # Hitung waktu saat ini (dalam detik) berdasarkan jumlah step environment
    t = env.episode_length_buf[env_ids] * env.step_dt
    
    # Konversi amplitudo dari derajat ke radian (90 derajat = ~1.57 rad)
    amp_rad = math.radians(amplitude_deg)
    
    # Hitung posisi target dengan fungsi sinusoidal
    # Osilasi akan bergerak mulus antara -amp_rad dan +amp_rad
    target_pos = amp_rad * torch.sin(2 * math.pi * freq_hz * t)
    
    # Cari indeks spesifik untuk joint l_sho_pitch dan r_sho_pitch di robot
    joint_indices, _ = asset.find_joints(asset_cfg.joint_names)
    
    if len(joint_indices) > 0:
        # Ambil target posisi persendian saat ini di memori
        current_targets = asset.data.joint_pos_target[env_ids].clone()
        
        # Eksekusi secara simultan: timpa target bahu dengan target sinusoidal
        target_pos_expanded = target_pos.unsqueeze(1).expand(-1, len(joint_indices))
        current_targets[:, joint_indices] = target_pos_expanded
        
        # Aplikasikan target posisi baru kembali ke robot
        asset.set_joint_position_target(current_targets, env_ids=env_ids)

def spoofed_joint_pos(env, asset_cfg: SceneEntityCfg):
    """Menyadap observasi posisi agar policy mengira upper body selalu statis di 0.0"""
    pos_rel = mdp.joint_pos_rel(env, asset_cfg).clone()
    asset = env.scene[asset_cfg.name]
    
    # Daftar semua joint upper body yang ingin disembunyikan pergerakannya dari policy
    joint_names_to_mask = [
        "l_sho_pitch", "r_sho_pitch", "head_pan", "head_tilt", 
        "left_j2", "right_j2", "l_sho_roll", "l_el", "l_wrist", 
        "r_sho_roll", "r_el", "r_wrist"
    ]
    joint_indices, _ = asset.find_joints(joint_names_to_mask)
    
    if len(joint_indices) > 0:
        # Paksa nilai observasi menjadi 0.0 khusus untuk joint upper body
        pos_rel[:, joint_indices] = 0.0
        
    return pos_rel

def spoofed_joint_vel(env, asset_cfg: SceneEntityCfg):
    """Menyadap observasi kecepatan sendi agar policy tidak melihat pergerakan lengan"""
    vel_rel = mdp.joint_vel_rel(env, asset_cfg).clone()
    asset = env.scene[asset_cfg.name]
    
    joint_names_to_mask = [
        "l_sho_pitch", "r_sho_pitch", "head_pan", "head_tilt", 
        "left_j2", "right_j2", "l_sho_roll", "l_el", "l_wrist", 
        "r_sho_roll", "r_el", "r_wrist"
    ]
    joint_indices, _ = asset.find_joints(joint_names_to_mask)
    
    if len(joint_indices) > 0:
        vel_rel[:, joint_indices] = 0.0
        
    return vel_rel
    
# ============================================================================
# OBSERVATION CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class PlayObservationsCfg(ObservationsCfg):
    """Mengganti observasi asli dengan fungsi spoofing kita"""
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        joint_pos = ObsTerm(func=spoofed_joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=spoofed_joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
        
# ============================================================================
# EVENT CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class PlayEventCfg:
    """
    Event configuration for play mode.
    Resets the scene and applies continuous shoulder oscillation.
    """
    
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Event robustness test yang dieksekusi terus-menerus
    swing_shoulders = EventTerm(
        func=swing_shoulders_event,
        mode="interval",  
        interval_range_s=(0.0, 0.0), # Interval 0.0 memastikan dieksekusi di setiap step
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_sho_pitch", "r_sho_pitch"]),
            "amplitude_deg": 90.0, # Bergerak hingga 90 dan -90 derajat
            "freq_hz": 0.5,        # 0.5 siklus per detik (1 ayunan penuh butuh 2 detik)
        }
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
        rel_standing_envs=0.0,    
        rel_heading_envs=0.0,
        ranges=vel_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),    
            lin_vel_y=(0.0, 0.0),    
            ang_vel_z=(0.0, 0.0),    
            heading=(0.0, 0.0),
        ),
        resampling_time_range=(100.0, 100.0), 
        debug_vis=True  
    )


# ============================================================================
# MAIN PLAY ENVIRONMENT CONFIGURATION
# ============================================================================
@configclass
class CentaurPlayEnvHandCfg(CentaurBalancingEnvCfg):
    events: PlayEventCfg = PlayEventCfg()
    commands: PlayCommandsCfg = PlayCommandsCfg()
    
    # Terapkan observasi manipulasi yang baru dibuat ke environment ini
    observations: PlayObservationsCfg = PlayObservationsCfg()
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.env_spacing = 4.0
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.episode_length_s = 30
