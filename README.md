# NXP Diablo – Reinforcement Learning for Self-Balancing and Velocity Control

This repository contains Deep Reinforcement Learning (RRL) training and evaluation code for **self-balancing** and **velocity-based locomotion control** of the **Diablo robot**, implemented using **NVIDIA Isaac Lab** and **RSL-RL (PPO)**.

The project focuses on achieving stable balancing behavior and controlled velocity movement through simulation-based reinforcement learning.

---

## 📁 Project Structure

```
NXP-Diablo/
├── Part_gripper_col_rev/
├── agents/
├── centaur2_new_visual.usd
├── centaur2_v3_env_cfg.py
├── centaur2_v3_env_cfg_play.py
└── model/
```

---

## 📂 Folder and File Description

### `Part_gripper_col_rev/`
Contains all robot parts used in the simulation, including:
- Solidworks parts
- STL mesh files
- Inertia values generated based on the robot’s real-world mass properties

### `agents/`
Contains reinforcement learning agent configurations.  
In this project, **rsl_rl_ppo_cfg** is used to achieve more stable training results using PPO (Proximal Policy Optimization).

### `centaur2_new_visual.usd`
The robot model in USD format, used as the main asset during reinforcement learning training and simulation.

### `centaur2_v3_env_cfg.py`
The main Isaac Lab environment configuration file for reinforcement learning training, including:
- Observation and action space
- Reward design
- Physics and control parameters

### `centaur2_v3_env_cfg_play.py`
Environment configuration used for running and evaluating the trained reinforcement learning model.

### `model/`
Contains trained reinforcement learning models.  
Use **model_4999.pt** as the final checkpoint for testing and evaluation.

---

## 🔧 Requirements

- Ubuntu (recommended)
- NVIDIA GPU with CUDA support
- NVIDIA Isaac Lab
- Python environment compatible with Isaac Lab
- RSL-RL (PPO)

---

## 📥 Clone the Repository

```bash
git clone https://github.com/NXP-Robots-Base-Hardware-Models/NXP-Diablo.git
cd NXP-Diablo
```

---

## 🚀 Training the Reinforcement Learning Model

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Centaur-Balancing2-v64 \
  --num_envs 1024 \
  --headless
```

---

## ▶️ Testing the Trained Model

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Centaur-Balancing2-v6-Play \
  --checkpoint path_to_model_folder/model_4999.pt \
  --num_envs 10
```

Replace `path_to_model_folder` with the actual path to the `model` directory.

---

## 🎯 Objective

- Achieve stable self-balancing behavior
- Enable controlled velocity movement
- Validate reinforcement learning performance in a physics-accurate simulation environment

---

## 📌 Notes

This repository is intended for research and development purposes.  
Simulation parameters and reward functions may require tuning depending on the robot configuration.

Damping/Stiffness Value Used in Training:

head_pan 0.00643/16.08461

head_tilt 0.00046/1.1578

l_sho_pitch 0.3/15

l_sho_roll 0.4/20

l_el 0.1/13.82181  

l_wrist 0.00182/4.54741

left_j4 25.0/120.0

left_j1 18.0/80.0      

left_j2 0.8/2.0   

left_j3 20/0.0

r_sho_pitch 0.3/15

r_sho_roll 0.4/20

r_el 0.1/13.82181     

r_wrist 0.00182/4.54741

right_j4 25.0/120.0

right_j1 18.0/80.0  

right_j2 0.8/2.0    

right_j3 20/0.0

