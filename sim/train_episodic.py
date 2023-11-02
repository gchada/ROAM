#! /usr/bin/env python
import os
import pickle

import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBufferEpisodic, ReplayBufferTrainingEpisodic

from env_utils import make_mujoco_env
from task_config_util import apply_task_configs
import rail_walker_gym

from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_float('upper_height', None, 'upper hurdle height. (A1)')
flags.DEFINE_float('lower_height', None, 'lower hurdle height. (A1)')
flags.DEFINE_string('hurdle', 'high', 'low or high hurdle. (A1)')
flags.DEFINE_string("algo_type", "vanilla_low", "algo name.")
flags.DEFINE_float("offline_ratio", 0, "Offline ratio.")
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", True, "Save agent replay buffer on evaluation."
)
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1000),
                     'Number of training steps to start training.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('load_buffer', False, 'Load replay buffer.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# ==================== Eval Flags ====================
flags.DEFINE_string("eval_env_name", "", "Environment name for evaluation. If empty, use training environment name.")
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')

# ==================== Log / Save Flags ====================
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer("save_interval", 10000, "Save interval.")
flags.DEFINE_string("save_dir", "./saved", "Directory to save the model checkpoint and replay buffer.")
flags.DEFINE_boolean('save_buffer', True, 'Save replay buffer for future training.')
flags.DEFINE_boolean('save_old_buffers', False, 'Keep replay buffers in previous steps.')
flags.DEFINE_string('project_name', 'go1-damage', 'wandb project name.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

flags.DEFINE_boolean('save_eval_videos', False, 'Save videos during evaluation.')
flags.DEFINE_integer("eval_video_length_limit", 0, "Limit the length of evaluation videos.")
flags.DEFINE_boolean('save_eval_rollouts', False, 'Save rollouts during evaluation.')
flags.DEFINE_boolean('save_training_videos', False, 'Save videos during training.')
flags.DEFINE_integer("training_video_length_limit", 0, "Limit the length of training videos.")
flags.DEFINE_integer("training_video_interval", 3000, "Interval to save training videos.")
flags.DEFINE_boolean('save_training_rollouts', False, 'Save rollouts during training.')

flags.DEFINE_boolean('launch_viewer', False, "Launch a windowed viewer for the off-screen rendered environment frames.")
flags.DEFINE_boolean('launch_target_viewer', False, "Launch a windowed viewer for joystick heading and target.")
# ========================================================

# ==================== Joystick Task Flags ====================
config_flags.DEFINE_config_file(
    'task_config',
    'task_configs/default.py',
    'File path to the task/control config parameters.',
    lock_config=False)
config_flags.DEFINE_config_file(
    'reset_agent_config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    exp_name = FLAGS.env_name
    wandb.init(project=FLAGS.project_name, name=exp_name)
    wandb.config.update(FLAGS)

    env = make_mujoco_env(FLAGS.env_name, upper_height=FLAGS.upper_height, lower_height=FLAGS.lower_height, hurdle=FLAGS.hurdle)
    task_suffix, env = apply_task_configs(env, FLAGS.env_name, FLAGS.max_steps, FLAGS.task_config, FLAGS.reset_agent_config, False)
    exp_name += task_suffix
    wandb.run.name = exp_name
    chkpt_dir = f'saved/checkpoints_{FLAGS.upper_height}_{FLAGS.lower_height}_{FLAGS.hurdle}_{FLAGS.algo_type}_{exp_name}'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f'saved/buffers_{FLAGS.upper_height}_{FLAGS.lower_height}_{FLAGS.hurdle}_{FLAGS.algo_type}_{exp_name}'
    os.makedirs(buffer_dir, exist_ok=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env = gym.wrappers.RecordVideo(
        env,
        f'videos/train_walk_{FLAGS.upper_height}_{FLAGS.lower_height}_{FLAGS.hurdle}_{FLAGS.algo_type}_{exp_name}',
        episode_trigger=lambda x: True)
    if FLAGS.save_training_videos:
        env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(env, record_every_n_steps=FLAGS.training_video_interval, video_length_limit=FLAGS.training_video_length_limit) # wrap environment to automatically save video to wandb
        env.enableWandbVideo = True

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")

    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBufferTrainingEpisodic(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    replay_buffer_final = ReplayBufferEpisodic(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer_final.seed(FLAGS.seed)

    observation, reset_info = env.reset(return_info=True, seed=FLAGS.seed, options=None)
    done = False
    success_step = 50000
    reset_steps = [0]
    done_count = 0
    accumulated_info_dict = {}
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        reward_f = env.task.reward_final

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                rewards_f=reward_f,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )

        replay_buffer_final.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward_f,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )

        observation = next_observation

        x, y, z = env.task.robot.get_3d_location()
        wandb.log({"X Position": x}, step=i + FLAGS.pretrain_steps)
        wandb.log({"Y Position": y}, step=i + FLAGS.pretrain_steps)
        wandb.log({"Z Position": z}, step=i + FLAGS.pretrain_steps)

        if done:
            observation, reset_info = env.reset(return_info=True, seed=FLAGS.seed, options=None)
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        for key in info.keys():
            if key in ['TimeLimit.truncated', 'TimeLimit.joystick_target_change', 'episode']:
                continue
            value = info[key]
            if key not in accumulated_info_dict:
                accumulated_info_dict[key] = [value]
            else:
                accumulated_info_dict[key].append(value)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio)
            )

            agent, update_info = agent.update(online_batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

                for k in accumulated_info_dict.keys():
                    v = accumulated_info_dict[k]
                    if v is None or len(v) <= 0:
                        continue

                    if k in ['fall_count', 'traversible_finished_lap_count']:
                        to_log = v[-1]
                    else:
                        to_log = np.mean(v)
                    wandb.log({'training/' + str(k): to_log}, step=i)
                accumulated_info_dict = {}

        if i % FLAGS.eval_interval == 0:

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f)
                    with open(os.path.join(buffer_dir, f"buffer_reward_final"), "wb") as f:
                        pickle.dump(replay_buffer_final, f)
                except:
                    print("Could not save agent buffer.")
    wandb.log({"Dones": done_count})
    wandb.log({"Success Step": success_step})

if __name__ == "__main__":
    app.run(main)
