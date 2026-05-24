# __init__.py
# Copyright (c) 2022-2024, The Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Self-balancing Centaur environment registry.
- Exposes a Gymnasium ID you can pass to your RL runners.
- Keeps the same ManagerBasedRLEnv entry_point used in Isaac Lab.
"""

import gymnasium as gym
from . import agents
from .centaur2_v3_env_cfg import CentaurBalancingEnvCfg
from .centaur2_v3_env_cfg_play import CentaurPlayEnvCfg
from .centaur2_v3_env_cfg_play_force import CentaurPlayEnvForceCfg
from .centaur2_v3_env_cfg_play_uneven import CentaurPlayEnvUnevenCfg
from .centaur2_v3_env_cfg_play_sho_pitch import CentaurPlayEnvHandCfg
from .centaur2_v3_env_cfg_play_square import CentaurPlayEnvSquareCfg
from .centaur2_v3_env_cfg_play_circular import CentaurPlayEnvCircularCfg
from .centaur2_v3_env_cfg_play_inversed_sho_pitch import CentaurPlayEnvHandInversedCfg

gym.register(
    id="Centaur-Balancing2-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Your env config class (python object)
        "env_cfg_entry_point": CentaurBalancingEnvCfg,
        # Choose ONE RL stack you’ll actually use. You can keep all if you like.
        # Point these to your agent configs (you can reuse your wheeled-quadruped ones at first).
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Centaur-Balancing2-v6-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)

gym.register(
    id="Centaur-Balancing2-v6-Play-Force",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvForceCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)

gym.register(
    id="Centaur-Balancing2-v6-Play-Uneven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvUnevenCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)
gym.register(
    id="Centaur-Balancing2-v6-Play-Hand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvHandCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)
gym.register(
    id="Centaur-Balancing2-v6-Play-Square",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvSquareCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)
gym.register(
    id="Centaur-Balancing2-v6-Play-Circular",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvCircularCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)
gym.register(
    id="Centaur-Balancing2-v6-Play-HandInversed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CentaurPlayEnvHandInversedCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CentaurBalancingPPORunnerCfg",
    },
)
