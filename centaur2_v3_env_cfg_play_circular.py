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
import math

# ============================================================================
# EVENT CONFIGURATION FOR PLAY MODE
# ============================================================================
@configclass
class PlayEventCfg:
    """
    Event configuration for play mode.
    Only resets the scene - does NOT modify commands.
    This allows you to set exact velocities without interference.
    """
    
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # NOTE: make_commands_exclusive is intentionally REMOVED


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
class CentaurPlayEnvCircularCfg(CentaurBalancingEnvCfg):
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
        self.viewer.eye = (0.0, 0.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Longer episodes for manual testing
        self.episode_length_s = 60  # 30 seconds instead of 12
