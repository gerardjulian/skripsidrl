"""
Play/Test configuration for Centaur robot.
This config removes the exclusive command logic so you can manually control velocities.

Usage:
    python play.py --task=Centaur-Balancing-v10-Play --num_envs=16
"""

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

# ============================================================================
# IMPORT UNTUK TERRAIN
# ============================================================================
import copy
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

# ============================================================================
# EVENT CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class PlayEventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


# ============================================================================
# COMMAND CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass  
class PlayCommandsCfg:
    velocity = vel_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        ranges=vel_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),    
            lin_vel_y=(0.0, 0.0),    
            ang_vel_z=(0.0, 0.0),    
            heading=(0.0, 0.0),
        ),
        resampling_time_range=(100.0, 100.0), 
        debug_vis=True  
    )


CUSTOM_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(60.0, 60.0),          # Ukuran per petak terrain
    border_width=20.0,        # Batas pinggir dunia
    num_rows=1,               # Jumlah baris (4x4 = 16 envs)
    num_cols=1,               # Jumlah kolom
    horizontal_scale=0.1,     # Resolusi horizontal grid
    vertical_scale=0.005,     # Resolusi vertikal grid
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,         # Matikan curriculum agar acak
    sub_terrains={
        # KITA HANYA MENDEFINISIKAN SATU JENIS TERRAIN SAJA DI SINI:
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,  # 100% grid akan diisi oleh medan berbatu ini
            
            # ==========================================================
            # ATUR KEKASARAN DI SINI (dalam meter):
            noise_range=(0.01, 0.05),  # (Min, Max) elevasi batu
            noise_step=0.01,           # Jarak antar batu / detail noise
            # ==========================================================
            border_width=0.25
        ),
    },
)

# ============================================================================
# SCENE CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class CentaurPlaySceneCfg(CentaurSceneCfg):
    """
    Overrides the original scene to purposefully inject SPECIFIC rough terrain.
    """
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CUSTOM_TERRAIN_CFG,
        max_init_terrain_level=5, 
    )
    
# ============================================================================
# MAIN PLAY ENVIRONMENT CONFIGURATION
# ============================================================================
@configclass
class CentaurPlayEnvUnevenCfg(CentaurBalancingEnvCfg):
    """
    Play configuration for Centaur.
    """
    scene: CentaurPlaySceneCfg = CentaurPlaySceneCfg()
    events: PlayEventCfg = PlayEventCfg()
    commands: PlayCommandsCfg = PlayCommandsCfg()
    
    def __post_init__(self) -> None:
        super().__post_init__()
        
        self.scene.env_spacing = 4.0
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.episode_length_s = 30
