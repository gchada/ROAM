# ROAM
This repository contains the code for the paper [ROAM]().

# Simulation:
Go to ```sim``` to run the following example commands.
## Stiffness Task
### ROAM: 
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_roam.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=stiffness --finetune=True --reg_weight=1.0
```
### HLC:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_hlc.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=stiffness --finetune=True
```
### RLPD FT:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_rlpd.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=friction --finetune=True
```

## Friction Task
### ROAM: 
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_roam.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=friction --finetune=True --reg_weight=100.0
```
### HLC:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_hlc.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=friction --finetune=True
```
### RLPD FT:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_rlpd.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=500 --max_steps=10000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=10000 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=friction --finetune=True
```

## Pretraining:
High Joint Stiffness:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_episodic.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=1000 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=2000 --max_steps=250000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=False --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=600 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=FR_0
```
Low Foot Friction:
```
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_episodic.py --env_name="Go1SanityMujoco-Empty-SepRew-v0" --save_buffer=False --utd_ratio=20 --start_training=1000 --config=configs/droq_config.py --config.critic_layer_norm=True --save_eval_videos=False --eval_interval=1000 --save_training_videos=True --training_video_interval=2000 --max_steps=50000 --log_interval=1000 --save_interval=5000 --seed=0 --tqdm=True --save_dir=go1_sl_damage --task_config.action_interpolation=True --task_config.enable_reset_policy=False --task_config.Kp=40 --task_config.Kd=5 --task_config.limit_episode_length=600 --task_config.action_range=0.3 --task_config.rew_target_velocity=1.0 --task_config.frame_stack=0 --task_config.action_history=0 --task_config.center_init_action=True --task_config.rew_energy_penalty_weight=0.008 --task_config.rew_qpos_penalty_weight=10.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.6 --task_config.rew_roll_rate_penalty_factor=0.6 --task_config.rew_joint_diagonal_penalty_weight=0.1 --task_config.rew_joint_shoulder_penalty_weight=0.15 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.modification=FR_friction
```
# Real World:
Go to ```real/training``` to run the following example commands.
### ROAM: 
```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_roam.py --env_name="Go1SanityReal-Empty-SepRew-v0" --save_buffer=True --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --config.exterior_linear_c=0.0 --save_eval_videos=True --eval_interval=-1 --save_training_videos=False --training_video_interval=5000 --eval_episodes=1 --max_steps=40000 --log_interval=500 --save_interval=10000 --seed=0 --project_name=indoor_reset --tqdm=True --save_dir=indoor_reset --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=20 --task_config.Kd=1.0 --task_config.limit_episode_length=0 --task_config.action_range=0.35 --task_config.frame_stack=0 --task_config.action_history=1 --task_config.rew_target_velocity=1.5 --task_config.rew_energy_penalty_weight=0.0 --task_config.rew_qpos_penalty_weight=2.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.4 --task_config.rew_roll_rate_penalty_factor=0.2 --task_config.rew_joint_diagonal_penalty_weight=0.03 --task_config.rew_joint_shoulder_penalty_weight=0.00 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.center_init_action=True --task_config.rew_contact_reward_weight=2.0 --action_curriculum_steps=60000 --action_curriculum_start=1.0 --action_curriculum_end=1.0 --action_curriculum_linear=True --action_curriculum_exploration_eps=0.2 --task_config.filter_actions=8 --reset_curriculum=True --task_config.rew_smooth_change_in_tdy_steps=1 --threshold=1.5 --num_agents=1
```
### HLC:
```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_hlc.py --env_name="Go1SanityReal-Empty-SepRew-v0" --save_buffer=True --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --config.exterior_linear_c=0.0 --save_eval_videos=True --eval_interval=-1 --save_training_videos=False --training_video_interval=5000 --eval_episodes=1 --max_steps=40000 --log_interval=500 --save_interval=10000 --seed=0 --project_name=indoor_reset --tqdm=True --save_dir=indoor_reset --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=20 --task_config.Kd=1.0 --task_config.limit_episode_length=0 --task_config.action_range=0.35 --task_config.frame_stack=0 --task_config.action_history=1 --task_config.rew_target_velocity=1.5 --task_config.rew_energy_penalty_weight=0.0 --task_config.rew_qpos_penalty_weight=2.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.4 --task_config.rew_roll_rate_penalty_factor=0.2 --task_config.rew_joint_diagonal_penalty_weight=0.03 --task_config.rew_joint_shoulder_penalty_weight=0.00 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.center_init_action=True --task_config.rew_contact_reward_weight=2.0 --action_curriculum_steps=60000 --action_curriculum_start=1.0 --action_curriculum_end=1.0 --action_curriculum_linear=True --action_curriculum_exploration_eps=0.2 --task_config.filter_actions=8 --reset_curriculum=True --task_config.rew_smooth_change_in_tdy_steps=1 --threshold=1.5 --num_agents=1
```
### Deployment without Behavior Modulation:
```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name="Go1SanityReal-Empty-SepRew-v0" --save_buffer=True --utd_ratio=20 --start_training=0 --config=configs/droq_config.py --config.critic_layer_norm=True --config.exterior_linear_c=0.0 --save_eval_videos=True --eval_interval=-1 --save_training_videos=False --training_video_interval=5000 --eval_episodes=1 --max_steps=40000 --log_interval=500 --save_interval=10000 --seed=0 --project_name=indoor_reset --tqdm=True --save_dir=indoor_reset --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=20 --task_config.Kd=1.0 --task_config.limit_episode_length=0 --task_config.action_range=0.35 --task_config.frame_stack=0 --task_config.action_history=1 --task_config.rew_target_velocity=1.5 --task_config.rew_energy_penalty_weight=0.0 --task_config.rew_qpos_penalty_weight=2.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.4 --task_config.rew_roll_rate_penalty_factor=0.2 --task_config.rew_joint_diagonal_penalty_weight=0.03 --task_config.rew_joint_shoulder_penalty_weight=0.00 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.center_init_action=True --task_config.rew_contact_reward_weight=2.0 --action_curriculum_steps=60000 --action_curriculum_start=1.0 --action_curriculum_end=1.0 --action_curriculum_linear=True --action_curriculum_exploration_eps=0.2 --task_config.filter_actions=8 --reset_curriculum=True --task_config.rew_smooth_change_in_tdy_steps=1 --threshold=1.5 --num_agents=1
```
### Pretraining
Normal Walking:
```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name="Go1SanityReal-Empty-SepRew-v0" --save_buffer=True --utd_ratio=20 --start_training=1000 --config=configs/droq_config.py --config.critic_layer_norm=True --config.exterior_linear_c=0.0 --save_eval_videos=True --eval_interval=-1 --save_training_videos=False --training_video_interval=5000 --eval_episodes=1 --max_steps=40000 --log_interval=500 --save_interval=10000 --seed=0 --project_name=indoor_reset --tqdm=True --save_dir=indoor_reset --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=20 --task_config.Kd=1.0 --task_config.limit_episode_length=0 --task_config.action_range=0.35 --task_config.frame_stack=0 --task_config.action_history=1 --task_config.rew_target_velocity=1.5 --task_config.rew_energy_penalty_weight=0.0 --task_config.rew_qpos_penalty_weight=2.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.4 --task_config.rew_roll_rate_penalty_factor=0.2 --task_config.rew_joint_diagonal_penalty_weight=0.03 --task_config.rew_joint_shoulder_penalty_weight=0.00 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.center_init_action=True --task_config.rew_contact_reward_weight=2.0 --action_curriculum_steps=60000 --action_curriculum_start=1.0 --action_curriculum_end=1.0 --action_curriculum_linear=True --action_curriculum_exploration_eps=0.2 --task_config.filter_actions=8 --reset_curriculum=True --task_config.rew_smooth_change_in_tdy_steps=1 --threshold=1.5 --num_agents=1
```
Dropped-out Joint:
```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name="Go1SanityReal-Empty-SepRew-v0" --save_buffer=True --utd_ratio=20 --start_training=1000 --config=configs/droq_config.py --config.critic_layer_norm=True --config.exterior_linear_c=0.0 --save_eval_videos=True --eval_interval=-1 --save_training_videos=False --training_video_interval=5000 --eval_episodes=1 --max_steps=40000 --log_interval=500 --save_interval=10000 --seed=0 --project_name=indoor_reset --tqdm=True --save_dir=indoor_reset --task_config.action_interpolation=True --task_config.enable_reset_policy=True --task_config.Kp=20 --task_config.Kd=1.0 --task_config.limit_episode_length=0 --task_config.action_range=0.35 --task_config.frame_stack=0 --task_config.action_history=1 --task_config.rew_target_velocity=1.5 --task_config.rew_energy_penalty_weight=0.0 --task_config.rew_qpos_penalty_weight=2.0 --task_config.rew_smooth_torque_penalty_weight=0.005 --task_config.rew_pitch_rate_penalty_factor=0.4 --task_config.rew_roll_rate_penalty_factor=0.2 --task_config.rew_joint_diagonal_penalty_weight=0.03 --task_config.rew_joint_shoulder_penalty_weight=0.00 --task_config.rew_joint_acc_penalty_weight=0.0 --task_config.rew_joint_vel_penalty_weight=0.0 --task_config.center_init_action=True --task_config.rew_contact_reward_weight=2.0 --action_curriculum_steps=60000 --action_curriculum_start=1.0 --action_curriculum_end=1.0 --action_curriculum_linear=True --action_curriculum_exploration_eps=0.2 --task_config.filter_actions=8 --reset_curriculum=True --task_config.rew_smooth_change_in_tdy_steps=1 --threshold=1.5 --num_agents=1 --leg_dropout_step=1 --dropout_joint=2
```

## Dependencies
To install dependencies for sim and real, create a conda environment using the environment_droplet.yaml in the respective directories. Then install dmcgym and jax using the commands below.
### Sim:
```
conda env create -f sim/environment_droplet.yml
conda activate roam_sim
pip install git+https://github.com/ikostrikov/dmcgym.git
pip install jaxlib==0.4.2+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.2
```
### Real:
```
conda env create -f real/environment_droplet.yml
conda activate roam_real
pip install git+https://github.com/ikostrikov/dmcgym.git
pip install jaxlib==0.4.14+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.14
```