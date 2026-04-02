# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="isaaclab-a1-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_flat_sac_cfg.yaml",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml"
    },
)

gym.register(
    id="quadloco-a1-flat-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg_PLAY",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml"
    },
)

gym.register(
    id="quadloco-a1-rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml"
    },
)

gym.register(
    id="quadloco-a1-rough-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg_PLAY",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml"
    },
)
