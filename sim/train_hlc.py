#! /usr/bin/env python
import os
import pickle
import gym
import numpy as np
import jax.numpy as jnp
import jax
import tqdm
from absl import app, flags
import plotly.graph_objs as go

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from env_utils import make_mujoco_env
from rlpd.wrappers import wrap_gym

from rlpd.networks import MLP
from keras.utils.np_utils import to_categorical
import optax
import jax
from flax.training.train_state import TrainState

from task_config_util import apply_task_configs
import rail_walker_gym

FLAGS = flags.FLAGS
colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a', '#19d3f3', '#ff6692', '#b6e880', '#ff97ff', '#fecb52']
flags.DEFINE_float('upper_height', None, 'upper hurdle height. (A1)')
flags.DEFINE_float('lower_height', None, 'lower hurdle height. (A1)')
flags.DEFINE_string('hurdle', 'high', 'low or high hurdle. (A1)')
flags.DEFINE_boolean("rlpd", True, "Use rlpd instead of sac ft.")
flags.DEFINE_boolean("resets", True, "Use resets.")
flags.DEFINE_boolean("finetune", True, "Update agent online.")
flags.DEFINE_float("reg_weight", 1.0, "Regularization weight.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("pretrain_steps", 2000, "Number of offline updates.")
flags.DEFINE_boolean("finetune_with_reg", False, "Use regularization for finetuning as well.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", True, "Save agent replay buffer on evaluation."
)
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5000), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(0),
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


def combine_imbalanced(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0:v.shape[0]] = v
            tmp[v.shape[0]:] = other_dict[k]
            combined[k] = tmp

    return combined


def update_state_network(state_network, batch):

    def loss_fn(state_net_params):
        output = state_network.apply_fn(
            {"params": state_net_params}, batch["observations"]
        )

        categorical_labels = to_categorical(batch["labels"], num_classes=4)

        loss = optax.softmax_cross_entropy(output, categorical_labels)
        return loss.mean(), {"loss": loss.mean()}

    grads, info = jax.grad(loss_fn, has_aux=True)(state_network.params)
    state_network = state_network.apply_gradients(grads=grads)

    return state_network, info


def main(_):
    f_str = "_train" if FLAGS.finetune else "_zero_shot"
    exp_name = FLAGS.env_name
    wandb.init(project=FLAGS.project_name, name=exp_name, mode="offline")
    wandb.config.update(FLAGS)

    env = make_mujoco_env(FLAGS.env_name, upper_height=FLAGS.upper_height, lower_height=FLAGS.lower_height, hurdle=FLAGS.hurdle)
    task_suffix, env = apply_task_configs(env, FLAGS.env_name, FLAGS.max_steps, FLAGS.task_config, FLAGS.reset_agent_config, False)
    exp_name += task_suffix
    exp_name += f_str
    exp_name += f"_{FLAGS.seed}"
    exp_name += f"_{FLAGS.reg_weight}"
    wandb.run.name = exp_name

    chkpt_dir = f'saved/checkpoints_{exp_name}'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f'saved/buffers_{exp_name}'
    os.makedirs(buffer_dir, exist_ok=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env = gym.wrappers.RecordVideo(
        env,
        f'videos/train_walk_{exp_name}',
        episode_trigger=lambda x: True)
    env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(env, record_every_n_steps=FLAGS.training_video_interval,
                                                          video_length_limit=FLAGS.training_video_length_limit)  # wrap environment to automatically save video to wandb
    env.enableWandbVideo = True

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    # model_cls += FLAGS.algorithm_version
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    np.random.seed(FLAGS.seed)

    if env.task.joystick_policy.reward_provider.modification == "friction":
        NUM_AGENTS = 4
        GOAL_DIST = 5
        # load in prior data buffers
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_friction/buffer_reward_final", "rb") as f:
            fr_buffer = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FL_friction/buffer_reward_final", "rb") as f:
            fl_buffer = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RR_friction/buffer_reward_final", "rb") as f:
            rr_buffer = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_friction/buffer_reward_final", "rb") as f:
            rl_buffer = pickle.load(f)
        full_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, 40000 * NUM_AGENTS)
        full_offline_buffer.seed(FLAGS.seed)

        full_offline_buffer.add(fr_buffer, 10000, 0)
        full_offline_buffer.add(fl_buffer, 10000, 1)
        full_offline_buffer.add(rr_buffer, 10000, 2)
        full_offline_buffer.add(rl_buffer, 10000, 3)

        fr_online_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fr_online_buffer.seed(FLAGS.seed)

        fl_online_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fl_online_buffer.seed(FLAGS.seed)

        rr_online_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rr_online_buffer.seed(FLAGS.seed)

        rl_online_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rl_online_buffer.seed(FLAGS.seed)

        fr_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fr_offline_buffer.seed(FLAGS.seed)

        fl_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fl_offline_buffer.seed(FLAGS.seed)

        rr_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rr_offline_buffer.seed(FLAGS.seed)

        rl_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rl_offline_buffer.seed(FLAGS.seed)

        fr_offline_buffer.add(fr_buffer, 10000, 0)
        fl_offline_buffer.add(fl_buffer, 10000, 1)
        rr_offline_buffer.add(rr_buffer, 10000, 2)
        rl_offline_buffer.add(rl_buffer, 10000, 3)

        fr_chkpt = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_friction")
        fl_chkpt = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_friction")
        rr_chkpt = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_friction")
        rl_chkpt = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_friction")

        fr_agent = checkpoints.restore_checkpoint(fr_chkpt, agent)
        fl_agent = checkpoints.restore_checkpoint(fl_chkpt, agent)
        rr_agent = checkpoints.restore_checkpoint(rr_chkpt, agent)
        rl_agent = checkpoints.restore_checkpoint(rl_chkpt, agent)

        agents = []
        buffers = []
        online_buffers = []
        offline_buffers = []

        agents.append(fr_agent)
        agents.append(fl_agent)
        agents.append(rr_agent)
        agents.append(rl_agent)

        buffers.append(fr_buffer)
        buffers.append(fl_buffer)
        buffers.append(rr_buffer)
        buffers.append(rl_buffer)

        online_buffers.append(fr_online_buffer)
        online_buffers.append(fl_online_buffer)
        online_buffers.append(rr_online_buffer)
        online_buffers.append(rl_online_buffer)

        offline_buffers.append(fr_offline_buffer)
        offline_buffers.append(fl_offline_buffer)
        offline_buffers.append(rr_offline_buffer)
        offline_buffers.append(rl_offline_buffer)

    elif env.task.joystick_policy.reward_provider.modification == "stiffness":
        NUM_AGENTS = 9
        GOAL_DIST = 10
        # load in prior data buffers
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_0/buffer_reward_final", "rb") as f:
            fr_buffer_0 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_2/buffer_reward_final", "rb") as f:
            fr_buffer_2 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FL_0/buffer_reward_final", "rb") as f:
            fl_buffer_0 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FL_2/buffer_reward_final", "rb") as f:
            fl_buffer_2 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RR_0/buffer_reward_final", "rb") as f:
            rr_buffer_0 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RR_2/buffer_reward_final", "rb") as f:
            rr_buffer_2 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_0/buffer_reward_final", "rb") as f:
            rl_buffer_0 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_1/buffer_reward_final", "rb") as f:
            rl_buffer_1 = pickle.load(f)
        with open("saved/episodic/buffers_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_2/buffer_reward_final", "rb") as f:
            rl_buffer_2 = pickle.load(f)

        full_offline_buffer = ReplayBuffer(env.observation_space, env.action_space, 40000 * NUM_AGENTS)
        full_offline_buffer.seed(FLAGS.seed)

        full_offline_buffer.add(fr_buffer_0, 210000, 0)
        full_offline_buffer.add(fr_buffer_2, 210000, 1)
        full_offline_buffer.add(fl_buffer_0, 210000, 2)
        full_offline_buffer.add(fl_buffer_2, 210000, 3)
        full_offline_buffer.add(rr_buffer_0, 210000, 4)
        full_offline_buffer.add(rr_buffer_2, 210000, 5)
        full_offline_buffer.add(rl_buffer_0, 210000, 6)
        full_offline_buffer.add(rl_buffer_1, 210000, 7)
        full_offline_buffer.add(rl_buffer_2, 210000, 8)

        fr_online_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fr_online_buffer_0.seed(FLAGS.seed)

        fr_online_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fr_online_buffer_2.seed(FLAGS.seed)

        fl_online_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fl_online_buffer_0.seed(FLAGS.seed)

        fl_online_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        fl_online_buffer_2.seed(FLAGS.seed)

        rr_online_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rr_online_buffer_0.seed(FLAGS.seed)

        rr_online_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rr_online_buffer_2.seed(FLAGS.seed)

        rl_online_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rl_online_buffer_0.seed(FLAGS.seed)

        rl_online_buffer_1 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rl_online_buffer_1.seed(FLAGS.seed)

        rl_online_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
        rl_online_buffer_2.seed(FLAGS.seed)

        fr_offline_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fr_offline_buffer_0.seed(FLAGS.seed)

        fr_offline_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fr_offline_buffer_2.seed(FLAGS.seed)

        fl_offline_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fl_offline_buffer_0.seed(FLAGS.seed)

        fl_offline_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        fl_offline_buffer_2.seed(FLAGS.seed)

        rr_offline_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rr_offline_buffer_0.seed(FLAGS.seed)

        rr_offline_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rr_offline_buffer_2.seed(FLAGS.seed)

        rl_offline_buffer_0 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rl_offline_buffer_0.seed(FLAGS.seed)

        rl_offline_buffer_1 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rl_offline_buffer_1.seed(FLAGS.seed)

        rl_offline_buffer_2 = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps * 10)
        rl_offline_buffer_2.seed(FLAGS.seed)

        fr_offline_buffer_0.add(fr_buffer_0, 210000, 0)
        fr_offline_buffer_2.add(fr_buffer_2, 210000, 1)

        fl_offline_buffer_0.add(fl_buffer_0, 210000, 2)
        fl_offline_buffer_2.add(fl_buffer_2, 210000, 3)

        rr_offline_buffer_0.add(rr_buffer_0, 210000, 4)
        rr_offline_buffer_2.add(rr_buffer_2, 210000, 5)

        rl_offline_buffer_0.add(rl_buffer_0, 210000, 6)
        rl_offline_buffer_1.add(rl_buffer_1, 210000, 7)
        rl_offline_buffer_2.add(rl_buffer_2, 210000, 8)

        fr_chkpt_0 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_0")
        fr_chkpt_2 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FR_2")

        fl_chkpt_0 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FL_0")
        fl_chkpt_2 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_FL_2")

        rr_chkpt_0 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RR_0")
        rr_chkpt_2 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RR_2")

        rl_chkpt_0 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_0")
        rl_chkpt_1 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_1")
        rl_chkpt_2 = checkpoints.latest_checkpoint("saved/episodic/checkpoints_None_None_high_vanilla_low_Go1SanityMujoco-Empty-SepRew-v0_nlt_PD40.00,5.00_ai_ar0.30_fs0_ciqa_ppf0.60_tv1.00_avpf0.60,0.60_jspw1.00e-01,1.50e-01_qpw0.00e+00,0.00e+00_ep0.008_qpw10.000_stpw5.00e-03_mod_RL_2")

        fr_agent_0 = checkpoints.restore_checkpoint(fr_chkpt_0, agent)
        fr_agent_2 = checkpoints.restore_checkpoint(fr_chkpt_2, agent)

        fl_agent_0 = checkpoints.restore_checkpoint(fl_chkpt_0, agent)
        fl_agent_2 = checkpoints.restore_checkpoint(fl_chkpt_2, agent)

        rr_agent_0 = checkpoints.restore_checkpoint(rr_chkpt_0, agent)
        rr_agent_2 = checkpoints.restore_checkpoint(rr_chkpt_2, agent)

        rl_agent_0 = checkpoints.restore_checkpoint(rl_chkpt_0, agent)
        rl_agent_1 = checkpoints.restore_checkpoint(rl_chkpt_1, agent)
        rl_agent_2 = checkpoints.restore_checkpoint(rl_chkpt_2, agent)

        agents = []
        buffers = []
        online_buffers = []
        offline_buffers = []

        agents.append(fr_agent_0)
        agents.append(fr_agent_2)

        agents.append(fl_agent_0)
        agents.append(fl_agent_2)

        agents.append(rr_agent_0)
        agents.append(rr_agent_2)

        agents.append(rl_agent_0)
        agents.append(rl_agent_1)
        agents.append(rl_agent_2)

        buffers.append(fr_buffer_0)
        buffers.append(fr_buffer_2)

        buffers.append(fl_buffer_0)
        buffers.append(fl_buffer_2)

        buffers.append(rr_buffer_0)
        buffers.append(rr_buffer_2)

        buffers.append(rl_buffer_0)
        buffers.append(rl_buffer_1)
        buffers.append(rl_buffer_2)

        online_buffers.append(fr_online_buffer_0)
        online_buffers.append(fr_online_buffer_2)

        online_buffers.append(fl_online_buffer_0)
        online_buffers.append(fl_online_buffer_2)

        online_buffers.append(rr_online_buffer_0)
        online_buffers.append(rr_online_buffer_2)

        online_buffers.append(rl_online_buffer_0)
        online_buffers.append(rl_online_buffer_1)
        online_buffers.append(rl_online_buffer_2)

        offline_buffers.append(fr_offline_buffer_0)
        offline_buffers.append(fr_offline_buffer_2)

        offline_buffers.append(fl_offline_buffer_0)
        offline_buffers.append(fl_offline_buffer_2)

        offline_buffers.append(rr_offline_buffer_0)
        offline_buffers.append(rr_offline_buffer_2)

        offline_buffers.append(rl_offline_buffer_0)
        offline_buffers.append(rl_offline_buffer_1)
        offline_buffers.append(rl_offline_buffer_2)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, state_net_key = jax.random.split(rng, 2)

    state_network_def = MLP(hidden_dims=(256, 256, 256, NUM_AGENTS))

    state_net_params = state_network_def.init(state_net_key, env.observation_space.sample())["params"]
    tx = optax.adam(learning_rate=3e-4)
    state_network = TrainState.create(
        apply_fn=state_network_def.apply,
        params=state_net_params,
        tx=tx,
    )

    for i in tqdm.tqdm(range(FLAGS.pretrain_steps)):
        batch = full_offline_buffer.sample(FLAGS.batch_size)
        state_network, info = update_state_network(state_network, batch)
        if i % 1000 == 0:
            logits = state_network.apply_fn(
                {"params": state_network.params}, full_offline_buffer.dataset_dict["observations"]
            )
            def accuracy(logits, labels):
                return np.mean(np.argmax(logits, axis=-1) == labels)
            wandb.log({"state_accuracy": accuracy(logits, full_offline_buffer.dataset_dict["labels"])}, step=i)
            wandb.log({"state_loss": info["loss"]}, step=i)

    success_step = 10000  # default success step if agent never reaches goal
    policy, max_agent, env.task.policy = 0, 0, 0
    observation, reset_info = env.reset(return_info=True, seed=FLAGS.seed, options=None)
    accumulated_info_dict = {}
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):

        logits = state_network.apply_fn(
            {"params": state_network.params}, observation
        )
        max_agent = np.argmax(logits)

        action_agent = agents[max_agent]
        online_buffer = online_buffers[max_agent]
        env.task.policy = max_agent
        policy = max_agent

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, action_agent = action_agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        online_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
                labels=max_agent
            )
        )
        observation = next_observation

        x, y, z = env.task.robot.get_3d_location()

        dist = (x**2 + y**2)**0.5
        wandb.log({"statistics/X Position": x}, step=i + FLAGS.pretrain_steps)
        wandb.log({"statistics/Y Position": y}, step=i + FLAGS.pretrain_steps)
        wandb.log({"statistics/Z Position": z}, step=i + FLAGS.pretrain_steps)
        wandb.log({"statistics/Distance": dist}, step=i + FLAGS.pretrain_steps)
        wandb.log({"statistics/Policy": policy}, step=i + FLAGS.pretrain_steps)

        # single life is done if the agent reaches the success state (gets to the goal)
        if done:
            observation, reset_info = env.reset(return_info=True, seed=FLAGS.seed, options=None)

        if dist >= GOAL_DIST:
            success_step = i
            break

        for key in info.keys():
            if key in ['TimeLimit.truncated', 'TimeLimit.joystick_target_change', 'episode']:
                continue
            value = info[key]
            if key not in accumulated_info_dict:
                accumulated_info_dict[key] = [value]
            else:
                accumulated_info_dict[key].append(value)


        if i >= FLAGS.start_training:
            if FLAGS.finetune:
                batch = online_buffer.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio)
                )
                action_agent, update_info = action_agent.update(batch, int(FLAGS.utd_ratio))

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
                        chkpt_dir, action_agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        if FLAGS.rlpd:
                            pickle.dump(online_buffer, f)
                        else:
                            pickle.dump(replay_buffer, f)
                except:
                    print("Could not save agent buffer.")
    wandb.log({"Success Step": success_step})
    vid = wandb.Video(f'videos/train_walk_{exp_name}' + "/rl-video-episode-0.mp4", format='mp4')
    wandb.log({"video": vid})


if __name__ == "__main__":
    app.run(main)
