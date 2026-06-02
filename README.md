# Deep Reinforcement Learning for Self-Balancing and Velocity Control

This repository contains Deep Reinforcement Learning (DRL) training and evaluation code for **self-balancing** and **velocity-based locomotion control** of the **Diablo robot with humanoid upper-body modification**, implemented using **NVIDIA Isaac Lab** and **RSL-RL (PPO)**.

The project focuses on achieving stable balancing behavior and controlled velocity movement through simulation-based reinforcement learning.

---

## 📁 Project Structure

```
skripsidrl/
├── Part_gripper_col_rev/
├── Evaluation/
├── agents/
├── Model-Vx.pt/
├── Model-Vxyw.pt/
├── centaur2_new_visual.usd
├── centaur2_v3_env_cfg.py
├── centaur2_v3_env_cfg_play.py
├── centaur2_v3_env_cfg_play_circular.py
├── centaur2_v3_env_cfg_play_force.py
├── centaur2_v3_env_cfg_play_inversed_sho_pitch.py
├── centaur2_v3_env_cfg_play_sho_pitch.py
├── centaur2_v3_env_cfg_play_square.py
└── centaur2_v3_env_cfg_play_uneven.py

---
## 📂 Folder and File Description

### 🔹 Core Directories
* **`Part_gripper_col_rev/`**: Contains all robot parts used in the simulation, including Solidworks parts, STL mesh files, and inertia values generated based on the robot’s real-world mass properties.
* **`agents/`**: Contains reinforcement learning agent configurations. In this project, `rsl_rl_ppo_cfg` is used to achieve stable training results using PPO (Proximal Policy Optimization).
* **`Evaluation/`**: Dedicated folder for storing evaluation logs, performance metrics, and analytical data from model testing.
* **`model-Vx.pt/`**: Contains trained model specialized for linear velocity control in the X-axis ($V_x$).
* **`Model-Vxyw.pt/`**: Contains trained model capable of handling multi-directional velocity and angular control ($V_x, V_y, \omega_z$).

### 🔹 Simulation Assets & Training Config
* **`centaur2_new_visual.usd`**: The robot model in Universal Scene Description (USD) format, used as the main asset during reinforcement learning training and simulation.
* **`centaur2_v3_env_cfg.py`**: The main Isaac Lab environment configuration file for training, defining observation/action spaces, reward functions, and physics parameters.

### 🔹 Evaluation & Custom Play Configurations
These files are variations of the environment config used to test the trained model's robustness under specific challenges:
* **`centaur2_v3_env_cfg_play.py`**: Standard environment configuration for running and evaluating the model.
* **`centaur2_v3_env_cfg_play_circular.py`**: Testing setup for evaluating circular path tracking.
* **`centaur2_v3_env_cfg_play_force.py`**: Robustness evaluation under external push/force disturbances.
* **`centaur2_v3_env_cfg_play_square.py`**: Testing setup for tracking a square trajectory.
* **`centaur2_v3_env_cfg_play_uneven.py`**: Evaluation on rough, bumpy, or uneven terrain.
* **`centaur2_v3_env_cfg_play_sho_pitch.py`** & **`_inversed_sho_pitch.py`**: Custom testing configurations analyzing the effects of standard and inverted shoulder pitch movements on the robot's balance.

---

## 🔧 Requirements

- Ubuntu (recommended)
- NVIDIA GPU with CUDA support
- NVIDIA Isaac Lab
- Python environment compatible with Isaac Lab
- RSL-RL (PPO)

## 📥 Clone the Repository

```bash
git clone https://github.com/.git](https://github.com/gerardjulian/skripsidrl.git
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
  --checkpoint path_to_model_folder/Model-Vx.pt \
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

