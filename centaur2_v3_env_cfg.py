import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


CENTAUR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/centaur2_v3/centaur2_new_visual.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), 
        joint_pos={
            "head_pan": 0.0, "head_tilt": 0.0,
            "left_j1": -1.5, "left_j2": -3.124,
            "left_j3": 0.0,"left_j4":0.0,
            "l_sho_pitch": 0.0,
            "l_wrist":0.0,
            "l_sho_roll": 0.0, "l_el": -1.571,
            "right_j1": -1.5, "right_j2": -3.124,
            "right_j3": 0.0,"right_j4":0.0,
            "r_sho_pitch": 0.0,
            "r_sho_roll": 0.0, "r_el": -1.571,
            "r_wrist":0.0,
        }
    ),
    actuators={
        # Upper leg actuators (balancing joints)
        "lj4_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_j4"],
            effort_limit_sim=9.6,
            velocity_limit_sim=12.043,
            stiffness=120.0,      
            damping=25.0,          
        ),
        "rj4_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_j4"],
            effort_limit_sim=9.6,
            velocity_limit_sim=12.043,
            stiffness=120.0,    
            damping=25.0,          
        ),
        # Middle leg actuators (height/mode changing)
        "lj1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_j1"],
            effort_limit_sim=9.6,
            velocity_limit_sim=12.043,
            stiffness=80.0,      
            damping=18.0,          
        ),
        "rj1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_j1"],
            effort_limit_sim=9.6,
            velocity_limit_sim=12.043,
            stiffness=80.0,    
            damping=18.0,          
        ),
        
        # Wheel actuators (driving joints)
       
        "lj3_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_j3"],
            effort_limit_sim=9.6,
            velocity_limit_sim=23.04,
            stiffness=0.0,         
            damping=20.0,          
        ),
        "rj3_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_j3"],
            effort_limit_sim=9.6,
            velocity_limit_sim=23.04,
            stiffness=0.0,        
            damping=20.0,         
        ),       
    },
)
@configclass
class CentaurSceneCfg(InteractiveSceneCfg):
    """Configuration for a Centaur robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # centaur robot
    robot: ArticulationCfg = CENTAUR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # Wheel velocity control
    joint_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_j3", "right_j3"], 
        scale=1.0,
        # Add small deadzone to prevent drift from numerical errors
        use_default_offset=True
    )
    
    # Upper leg position control (balancing)
    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["left_j4","right_j4","left_j1","right_j1"], 
        scale=1.0,
        # Add small deadzone
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Base motion
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # Joint states - CRITICAL for detecting asymmetry
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Orientation for balance
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        
        # Commands and actions
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# ============================================================================
# EXCLUSIVE COMMAND FUNCTION
# ============================================================================
def make_velocity_commands_exclusive(env, env_ids, command_name="velocity", 
                                     prob_linear=0.5, prob_angular=0.30):
    """Post-process velocity commands to ensure only ONE type of motion at a time."""
    cmd_term = env.command_manager._terms[command_name]
    commands = cmd_term.command  # [num_envs, 3] -> [lin_x, lin_y, ang_z]
    
    if env_ids is None or len(env_ids) == 0:
        return
    
    rand = torch.rand(len(env_ids), device=env.device)
    prob_total = prob_linear + prob_angular
    prob_linear_norm = prob_linear / prob_total
    
    linear_mask = rand < prob_linear_norm
    angular_mask = ~linear_mask
    
    for i, env_id in enumerate(env_ids):
        if linear_mask[i]:
            commands[env_id, 2] = 0.0  # Zero turning
        else:
            commands[env_id, 0] = 0.0  # Zero forward/backward
            commands[env_id, 1] = 0.0  # Zero lateral



# ============================================================================
# HELPER FUNCTIONS FOR REWARDS
# ============================================================================
def _yaw_from_quat(q):
    x, y, z, w = q.unbind(dim=-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

def heading_cos(env, asset_cfg, use_random_ref=False):
    """cos(yaw - yaw0) if use_random_ref, else cos(yaw)."""
    root = env.scene[asset_cfg.name].data.root_state_w
    yaw = _yaw_from_quat(root[:, 3:7])
    if use_random_ref and hasattr(env, "episode_initial_yaw"):
        yaw0 = env.episode_initial_yaw
        return torch.cos(yaw - yaw0)
    return torch.cos(yaw)

def base_ang_vel_l2(env, asset_cfg):
    w = mdp.base_ang_vel(env, asset_cfg)
    return (w ** 2).sum(dim=1)

def vx_l2(env, asset_cfg):
    v = mdp.base_lin_vel(env, asset_cfg)
    vx = v[:, 0]
    return vx * vx

def vy_l2(env, asset_cfg):
    v = mdp.base_lin_vel(env, asset_cfg)
    vy = v[:, 1]
    return vy * vy

def vy_l2_scaled(env, asset_cfg):
    """Lateral velocity penalty scaled by forward speed."""
    v = mdp.base_lin_vel(env, asset_cfg)
    vx = torch.abs(v[:, 0])
    vy = v[:, 1]
    scale = 1.0 + 2.0 * vx
    return (vy * vy) * scale

def heading_alignment(env, asset_cfg):
    """Reward for keeping heading aligned with velocity direction."""
    v = mdp.base_lin_vel(env, asset_cfg)
    vx, vy = v[:, 0], v[:, 1]
    
    speed = torch.sqrt(vx**2 + vy**2)
    moving_mask = speed > 0.1
    
    alignment = torch.abs(vy) / (torch.abs(vx) + 1e-6)
    reward = torch.exp(-alignment**2)
    
    return torch.where(moving_mask, reward, torch.ones_like(reward))
    
def vx_track(env, asset_cfg, target_vx=0.3):
    v = mdp.base_lin_vel(env, asset_cfg)
    vx = v[:, 0]
    return torch.exp(-((vx - target_vx) ** 2) / (2 * (0.15 ** 2)))
    
def _get_pitch_rad(env, asset_cfg):
    base_ang = mdp.base_euler_xyz(env, asset_cfg)
    return base_ang[:, 1]

def _exp_track_1d(x, target, sigma_sq=0.25):
    return torch.exp(-((x - target) ** 2) / sigma_sq)

def track_lin_vel_x_exp_fixed(env, asset_cfg, target_vx=0.0):
    v = mdp.base_lin_vel(env, asset_cfg)
    return _exp_track_1d(v[:, 0], target_vx)

def track_yaw_rate_exp_fixed(env, asset_cfg, target_yaw_rate=0.0):
    w = mdp.base_ang_vel(env, asset_cfg)
    return _exp_track_1d(w[:, 2], target_yaw_rate)
    
def pitch_large_penalty(env, asset_cfg, threshold_sq=0.99):
    """Penalize only if pitch^2 >= threshold_sq."""
    root = env.scene[asset_cfg.name].data.root_state_w
    quat = root[:, 3:7]
    euler = quat_to_euler_xyz(quat)
    pitch = euler[:, 1]
    val = pitch * pitch
    mask = (val >= threshold_sq).float()
    return val * mask

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x,y,z,w) to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat.unbind(dim=-1)

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                        torch.sign(sinp) * (torch.pi / 2),
                        torch.asin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def torques_l2_sum(env, asset_cfg):
    """Sum of squared joint torques across all DOFs."""
    art = env.scene[asset_cfg.name]
    tau = getattr(art.data, "joint_torques", None)
    if tau is None or tau.numel() == 0:
        return mdp.action_l2(env)
    return (tau * tau).sum(dim=1)

def track_lin_vel_x_exp_cmd(env, asset_cfg, sigma_sq=0.25):
    v = mdp.base_lin_vel(env, asset_cfg)
    cmd = env.command_manager.get_command("velocity")
    return torch.exp(-((v[:, 0] - cmd[:, 0]) ** 2) / sigma_sq)

def track_lin_vel_y_exp_cmd(env, asset_cfg, sigma_sq=0.25):
    """Track commanded lateral velocity (should be zero)."""
    v = mdp.base_lin_vel(env, asset_cfg)
    cmd = env.command_manager.get_command("velocity")
    return torch.exp(-((v[:, 1] - cmd[:, 1]) ** 2) / sigma_sq)
    
def upright_posture_reward(env, asset_cfg):
    """Reward for keeping the robot upright (pitch and roll near zero)."""
    root = env.scene[asset_cfg.name].data.root_state_w
    quat = root[:, 3:7]
    euler = quat_to_euler_xyz(quat)
    roll = euler[:, 0]
    pitch = euler[:, 1]
    
    # Exponential reward peaks at zero tilt
    roll_reward = torch.exp(-(roll ** 2) / (2 * 0.1**2))  # sigma=0.1 rad (~5.7 deg)
    pitch_reward = torch.exp(-(pitch ** 2) / (2 * 0.1**2))
    
    return roll_reward * pitch_reward
    
def balancing_joints_centered(env, asset_cfg):
    """Reward for keeping L1/R1 joints near zero (upright stance)."""
    
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    joint_names = env.scene[asset_cfg.name].data.joint_names
    
    # Find indices for l1 and r1
    l1_idx = joint_names.index("l1") if "l1" in joint_names else None
    r1_idx = joint_names.index("r1") if "r1" in joint_names else None
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    if l1_idx is not None:
        penalty += (joint_pos[:, l1_idx] ** 2)
    if r1_idx is not None:
        penalty += (joint_pos[:, r1_idx] ** 2)
    
    # Convert to reward (penalize deviation from zero)
    return torch.exp(-penalty / (2 * 0.15**2))  # sigma=0.15 rad (~8.6 deg)  
    
def track_yaw_rate_exp_cmd(env, asset_cfg, sigma_sq=0.25):
    w = mdp.base_ang_vel(env, asset_cfg)
    cmd = env.command_manager.get_command("velocity")
    return torch.exp(-((w[:, 2] - cmd[:, 2]) ** 2) / sigma_sq)
    
    
# ============================================================================
# EVENT CONFIGURATION
# ============================================================================
@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    make_commands_exclusive = EventTerm(
        func=make_velocity_commands_exclusive,
        mode="reset",
        params={
            "command_name": "velocity",
            "prob_linear": 0.50,
            "prob_angular": 0.30
        }
    )


# ============================================================================
# REWARDS CONFIGURATION - Added symmetry penalties
# ============================================================================
@configclass
class RewardsCfg:
    """Reward terms for command tracking and stability."""

    # Don't drift sideways
    no_side_drift = RewTerm(
        func=vy_l2_scaled, weight=-1.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Keep heading aligned with velocity
    heading_align = RewTerm(
        func=heading_alignment, weight=0.7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Survival bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.7)

    # Track commanded forward/backward velocity
    lin_vel_x_track = RewTerm(
        func=track_lin_vel_x_exp_cmd, weight=+2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma_sq": 0.25},
    )
    
    # Track commanded lateral velocity
    lin_vel_y_track = RewTerm(
        func=track_lin_vel_y_exp_cmd, weight=+1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma_sq": 0.25},
    )
    
    # Track commanded turning rate
    yaw_rate_track = RewTerm(
        func=track_yaw_rate_exp_cmd, weight=+1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma_sq": 0.25},
    )

    upright_posture = RewTerm(
        func=upright_posture_reward, weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Pitch penalty
    pitch_penalty = RewTerm(
        func=pitch_large_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold_sq": 0.5},
    )
    
    joints_centered = RewTerm(
        func=balancing_joints_centered, weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    

# ============================================================================
# TERMINATIONS CONFIGURATION
# ============================================================================
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi / 3},
    )


# ============================================================================
# COMMANDS CONFIGURATION
# ============================================================================
@configclass
class CommandsCfg:
    """Configuration for velocity commands."""
    
    velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.6, 0.6),
            heading=(0.0, 0.0),
        ),
        resampling_time_range=(5.0, 8.0),
        debug_vis=True
    )


# ============================================================================
# CURRICULUM CONFIGURATION
# ============================================================================
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass


# ============================================================================
# MAIN ENVIRONMENT CONFIGURATION
# ============================================================================
@configclass
class CentaurBalancingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Centaur environment with drift fixes."""

    scene: CentaurSceneCfg = CentaurSceneCfg(num_envs=4096, env_spacing=4.0)
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.sim.dt = 0.005
        self.episode_length_s = 18
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.render_interval = self.decimation
