import numpy as np
from typing import Any, Optional
from rail_walker_interface import JoystickPolicyRewardProvider, JoystickPolicyTerminationConditionProvider
from rail_walker_interface import BaseWalker
from rail_mujoco_walker import RailSimWalkerDMControl
import transforms3d as tr3d
from dm_control.utils import rewards
import typing
from collections import deque

from rail_walker_interface.joystick_policy.joystick_interfaces import JoystickPolicyTerminationConditionProvider
from rail_walker_interface.robot.robot import BaseWalker
from .reward_util import near_quadratic_bound, calculate_gaussian_activation, calculate_torque

JOINT_WEIGHTS = np.array([1.0, 0.75, 0.5] * 4)


class JoystickPolicyStrictRewardProvider(JoystickPolicyRewardProvider[BaseWalker]):
    def __init__(
            self,
            energy_penalty_weight: float = 0.0,
            smooth_torque_penalty_weight: float = 0.0,
            joint_diagonal_penalty_weight: float = 0.0,
            joint_shoulder_penalty_weight: float = 0.0,
            joint_acc_penalty_weight: float = 0.0,
            joint_vel_penalty_weight: float = 0.0,
            pitch_rate_penalty_factor: float = 0.0,
            roll_rate_penalty_factor: float = 0.0,
            qpos_penalty_weight: float = 0.0,
            modification: str = "none"
    ) -> None:
        self.energy_penalty_weight = energy_penalty_weight
        self.smooth_torque_penalty_weight = smooth_torque_penalty_weight
        self.joint_diagonal_penalty_weight = joint_diagonal_penalty_weight
        self.joint_shoulder_penalty_weight = joint_shoulder_penalty_weight
        self.joint_acc_penalty_weight = joint_acc_penalty_weight
        self.joint_vel_penalty_weight = joint_vel_penalty_weight
        self.pitch_rate_penalty_factor = pitch_rate_penalty_factor
        self.roll_rate_penalty_factor = roll_rate_penalty_factor
        self.qpos_penalty_weight = qpos_penalty_weight
        self.modification = modification
        self.rew = 0.0
        self._last_torque = None

    def get_reward(self) -> float:
        return self.rew

    def get_reward_final(self) -> float:
        return self.rew_final

    def reset_reward(
            self,
            Robot: BaseWalker,
            info_dict: dict[str, Any],
            termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
            randomState: np.random.RandomState
    ) -> None:
        self.rew = 0.0
        self._last_torque = Robot.get_joint_torques().copy()

    def calculate_velocity_reward_norm(
            self,
            Robot: BaseWalker,
            action_target_qpos: np.ndarray,
            target_goal_world_delta: np.ndarray,
            target_goal_local: np.ndarray,
            target_yaw: float,
            target_delta_yaw: float,
            target_velocity: float,
            velocity_to_goal: float,
            change_in_abs_target_delta_yaw: float,
            target_custom_data: Optional[Any],
            enable_target_custom_obs: bool,
            info_dict: dict[str, Any],
            randomState: np.random.RandomState
    ) -> float:
        velocity_local = Robot.get_3d_local_velocity()
        roll, pitch, yaw = Robot.get_roll_pitch_yaw()

        # reward_v = rewards.tolerance(
        #     (np.cos(pitch) * velocity_local[0]),
        #     bounds=(target_velocity,
        #             target_velocity + 0.1),
        #     margin=target_velocity,
        #     value_at_margin=0,
        #     sigmoid='linear'
        # ) * target_velocity
        projected_x_velocity = np.cos(pitch) * velocity_local[0]
        reward_v = near_quadratic_bound(
            projected_x_velocity,
            target_velocity,
            target_velocity,
            target_velocity,
            None,  # "gaussian", #"linear",
            1.6,
            0.0
        )
        reward_v *= (1 + np.cos(target_delta_yaw)) / 2

        return reward_v

    def step_reward(
            self,
            Robot: BaseWalker,
            action_target_qpos: np.ndarray,
            target_goal_world_delta: np.ndarray,
            target_goal_local: np.ndarray,
            target_yaw: float,
            target_delta_yaw: float,
            target_velocity: float,
            velocity_to_goal: float,
            change_in_abs_target_delta_yaw: float,
            target_custom_data: Optional[Any],
            enable_target_custom_obs: bool,
            info_dict: dict[str, Any],
            randomState: np.random.RandomState
    ) -> None:
        velocity_norm = np.linalg.norm(Robot.get_3d_linear_velocity())

        if target_velocity >= 0.05:
            reward_v = self.calculate_velocity_reward_norm(
                Robot, action_target_qpos, target_goal_world_delta, target_goal_local, target_yaw, target_delta_yaw,
                target_velocity, velocity_to_goal, change_in_abs_target_delta_yaw, target_custom_data,
                enable_target_custom_obs, info_dict, randomState
            )
            reward_v *= max(target_velocity,
                            1 / 5)  # Scale so that the reward is more consistent with same velocity but different target velocities
            reward_v *= 15
            step_energy_coefficeint = self.energy_penalty_weight
        else:
            velocity_global = Robot.get_3d_linear_velocity()
            reward_close_to_0 = 1 - np.linalg.norm(velocity_global) * 5  # Encourage velocity to be close to 0
            reward_v = reward_close_to_0 * 3
            step_energy_coefficeint = 0.0

        info_dict["reward_v"] = reward_v

        # Calculate upright coefficient
        up = tr3d.quaternions.quat2mat(Robot.get_framequat_wijk())[-1, -1]  # See if the z-axis is pointing up

        # upright_coefficient = rewards.tolerance(up,
        #                             bounds=(0.9, float('inf')),
        #                             sigmoid='quadratic',
        #                             margin=0.9,
        #                             value_at_margin=0)
        upright_coefficient = (0.5 * up + 0.5) ** 2
        info_dict["reward_upright_coefficient"] = upright_coefficient

        reward_v_constrained = reward_v * upright_coefficient
        info_dict["reward_v_constrained"] = reward_v_constrained

        # Calculate Qpos Reward
        if enable_target_custom_obs and target_custom_data is not None and isinstance(target_custom_data,
                                                                                      dict) and "should_crouch" in target_custom_data.keys() and \
                target_custom_data["should_crouch"]:
            tar_pose = Robot.joint_qpos_crouch
        else:
            tar_pose = Robot.joint_qpos_init

        offset = Robot.joint_qpos_offset
        bounds = (tar_pose - offset, tar_pose + offset)

        qpos = Robot.get_joint_qpos()

        qpos_penalty_norm = 1.0
        for i in range(len(qpos)):
            single_qpos_rew = rewards.tolerance(
                qpos[i],
                bounds=(bounds[0][i], bounds[1][i]),
                margin=offset[i],
                value_at_margin=0.6
            )
            qpos_penalty_norm *= single_qpos_rew
        #####################################
        info_dict["penalty_qpos_normalized"] = qpos_penalty_norm
        qpos_penalty = (1.0 - qpos_penalty_norm) * self.qpos_penalty_weight
        info_dict["penalty_qpos"] = qpos_penalty

        # shifted_qpos = np.concatenate((
        #     qpos[0:3], [-qpos[3]], qpos[4:6], qpos[6:9], [-qpos[9]], qpos[10:]
        # ))
        # qpos_by_fr = np.stack(
        #     (
        #         np.stack((shifted_qpos[0:3], shifted_qpos[3:6]), axis=0),
        #         np.stack((shifted_qpos[6:9], shifted_qpos[9:12]), axis=0)
        #     ),
        #     axis=0
        # )
        # avg_qpos_by_fr = np.mean(qpos_by_fr, axis=1).flatten()
        # real_bounds = []
        # for bound in bounds:
        #     real_bound = np.concatenate((
        #         bound[0:3], bound[6:9]
        #     ))
        #     real_bounds.append(real_bound)
        # real_offset = np.concatenate((
        #     offset[0:3], offset[6:9]
        # ))
        # qpos_penalty_norm = 1.0
        # for i in range(len(avg_qpos_by_fr)):
        #     single_qpos_rew = rewards.tolerance(
        #         avg_qpos_by_fr[i],
        #         bounds=(real_bounds[0][i], real_bounds[1][i]),
        #         margin=real_offset[i],
        #         value_at_margin=0.1
        #     )
        #     qpos_penalty_norm *= single_qpos_rew
        # #####################################
        # info_dict["penalty_qpos_normalized"] = qpos_penalty_norm
        # qpos_penalty = (1.0-qpos_penalty_norm) * self.qpos_penalty_weight
        # info_dict["penalty_qpos"] = qpos_penalty

        # Angular velocity penalty
        angular_velocity = Robot.get_3d_angular_velocity()
        pitch_rate_penalty = self.pitch_rate_penalty_factor * (np.abs(angular_velocity[1]) ** 1.4) - 0.5
        roll_rate_penalty = self.roll_rate_penalty_factor * (np.abs(angular_velocity[0]) ** 1.4) - 0.5
        info_dict["pitch_rate_penalty"] = pitch_rate_penalty
        info_dict["roll_rate_penalty"] = roll_rate_penalty

        # Calculate Energy / Qvel Penalty
        joint_qvel = Robot.get_joint_qvel()
        joint_torques = Robot.get_joint_torques()
        energy = np.sum(np.abs(joint_qvel * joint_torques))
        # energy_sq = np.mean(np.abs(qvel) * (np.abs(torque) ** 1.5))

        if step_energy_coefficeint > 0.0:
            # energy_penalty = -0.5 * np.exp(-0.1*energy)
            energy_penalty = step_energy_coefficeint * (energy - 50 * (2 ** (Robot.limit_action_range / 0.3 - 1)))
            # energy_penalty = step_energy_coefficeint * (np.mean(np.abs(Robot.get_joint_torques()) ** 1.5) - 15)
            # energy_penalty = 0.04 * energy_sq - 1
            # energy_penalty = 0.02 * np.mean((np.abs(torque) ** 1.3)) - 1
        else:
            energy_penalty = 0  # -np.exp(-0.1*np.linalg.norm(qvel))

        info_dict['energy'] = energy
        # info_dict["energy_sq"] = energy_sq
        info_dict['penalty_energy'] = energy_penalty

        # Calculate Smooth Torque Penalty
        pd_joint_torques = calculate_torque(
            Robot.get_joint_qpos(),
            Robot.get_joint_qvel(),
            action_target_qpos,
            Robot.Kp,
            Robot.Kd
        )
        diff_torque = (pd_joint_torques - self._last_torque)
        diff_torque_norm = np.linalg.norm(diff_torque) ** 1.5

        smooth_torque_penalty = self.smooth_torque_penalty_weight * (
                    diff_torque_norm - 150) if self.smooth_torque_penalty_weight > 0 else 0.0
        info_dict["smooth_torque_penalty"] = smooth_torque_penalty
        info_dict["diff_torque_norm"] = diff_torque_norm

        self._last_torque = pd_joint_torques.copy()

        # Calculate joint diagonal / shoulder symmetry penalty
        FR_qvel = joint_qvel[0:3]
        FL_qvel = joint_qvel[3:6]
        FL_qvel[0] *= -1
        RR_qvel = joint_qvel[6:9]
        RL_qvel = joint_qvel[9:12]
        RL_qvel[0] *= -1
        diagonal_difference = np.linalg.norm(FR_qvel[1:] - RL_qvel[1:]) + np.linalg.norm(FL_qvel[1:] - RR_qvel[1:])
        shoulder_difference = np.linalg.norm(FR_qvel[1:] - FL_qvel[1:]) + 2 * np.linalg.norm(
            RR_qvel[1:] - RL_qvel[1:])  # penalize harder on the back legs
        shoulder_center = np.linalg.norm(FR_qvel[1:] + FL_qvel[1:]) + 2 * np.linalg.norm(RR_qvel[1:] + RL_qvel[1:])
        diagonal_difference_penalty = self.joint_diagonal_penalty_weight * diagonal_difference
        diagonal_difference_penalty *= (0.5 + velocity_norm) / (1.0 + 4.0 * np.abs(target_delta_yaw) / np.pi)

        shoulder_difference_penalty = self.joint_shoulder_penalty_weight * -(shoulder_difference)
        shoulder_difference_penalty *= (0.5 + velocity_norm) / (1.0 + 0.5 * np.abs(target_delta_yaw) / np.pi)

        info_dict["diagonal_difference_penalty"] = diagonal_difference_penalty
        info_dict["shoulder_difference_penalty"] = shoulder_difference_penalty

        # Calculate joint acceleration penalty
        joint_qacc = Robot.get_joint_qacc()
        joint_qacc_penalty = self.joint_acc_penalty_weight * np.linalg.norm(joint_qacc, ord=1)
        info_dict["joint_qacc_penalty"] = joint_qacc_penalty

        # Calculate joint velocity penalty
        joint_qvel_penalty = self.joint_vel_penalty_weight * np.linalg.norm(joint_qvel)
        info_dict["joint_qvel_penalty"] = joint_qvel_penalty

        z = Robot.get_3d_location()[2]
        if self.modification == "low":
            reward_z = rewards.tolerance(z,
                                         bounds=(0.37, 0.39),
                                         margin=0.1,
                                         value_at_margin=0,
                                         sigmoid='linear')
        elif self.modification == "high":
            reward_z = rewards.tolerance(z,
                                         bounds=(0.13, 0.15),
                                         margin=0.1,
                                         value_at_margin=0,
                                         sigmoid='linear')
        else:
            reward_z = 1.0

        rew_perstep = reward_v_constrained * reward_z - qpos_penalty - energy_penalty - smooth_torque_penalty - roll_rate_penalty - pitch_rate_penalty - diagonal_difference_penalty - shoulder_difference_penalty - joint_qacc_penalty - joint_qvel_penalty
        rew_perstep_final = reward_v_constrained - qpos_penalty - energy_penalty - smooth_torque_penalty - roll_rate_penalty - pitch_rate_penalty - diagonal_difference_penalty - shoulder_difference_penalty - joint_qacc_penalty - joint_qvel_penalty

        self.rew = rew_perstep
        self.rew_final = rew_perstep_final


class JoystickPolicySeperateRewardProvider(JoystickPolicyStrictRewardProvider):
    def __init__(self, smooth_tdy_steps: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_tdy = deque(maxlen=smooth_tdy_steps)

    @property
    def smooth_tdy_steps(self) -> int:
        return self.queue_tdy.maxlen

    @smooth_tdy_steps.setter
    def smooth_tdy_steps(self, value: int) -> None:
        self.queue_tdy = deque(maxlen=value)

    def calculate_velocity_reward_norm(
            self,
            Robot: BaseWalker,
            action_target_qpos: np.ndarray,
            target_goal_world_delta: np.ndarray,
            target_goal_local: np.ndarray,
            target_yaw: float,
            target_delta_yaw: float,
            target_velocity: float,
            velocity_to_goal: float,
            change_in_abs_target_delta_yaw: float,
            target_custom_data: Optional[Any],
            enable_target_custom_obs: bool,
            info_dict: dict[str, Any],
            randomState: np.random.RandomState
    ) -> float:
        reward_v = super().calculate_velocity_reward_norm(
            Robot,
            action_target_qpos,
            target_goal_world_delta,
            target_goal_local,
            target_yaw,
            target_delta_yaw,
            target_velocity,
            velocity_to_goal,
            change_in_abs_target_delta_yaw,
            target_custom_data,
            enable_target_custom_obs,
            info_dict,
            randomState
        )
        reward_v /= (1 + np.cos(target_delta_yaw)) / 2  # Revert the cosine reward
        return reward_v

    def step_reward(
            self,
            Robot: BaseWalker,
            action_target_qpos: np.ndarray,
            target_goal_world_delta: np.ndarray,
            target_goal_local: np.ndarray,
            target_yaw: float,
            target_delta_yaw: float,
            target_velocity: float,
            velocity_to_goal: float,
            change_in_abs_target_delta_yaw: float,
            target_custom_data: Optional[Any],
            enable_target_custom_obs: bool,
            info_dict: dict[str, Any],
            randomState: np.random.RandomState
    ) -> None:
        super().step_reward(
            Robot,
            action_target_qpos,
            target_goal_world_delta,
            target_goal_local,
            target_yaw,
            target_delta_yaw,
            target_velocity,
            velocity_to_goal,
            change_in_abs_target_delta_yaw,
            target_custom_data,
            enable_target_custom_obs,
            info_dict,
            randomState
        )
        self.queue_tdy.append(change_in_abs_target_delta_yaw)
        if target_velocity >= 0.05:
            rew_step = self.rew

            smoothed_change_in_tdy = np.mean(self.queue_tdy)

            # if target_delta_yaw != 0.0:
            target_change_tdy = 45.0 / 180.0 * np.pi * Robot.control_timestep * (target_delta_yaw / np.pi)
            target_change_tdy = -np.abs(target_change_tdy)

            # else:
            #     target_change_tdy = 0.0

            info_dict["target_change_tdy"] = target_change_tdy

            target_change_tdy_left_slack = 45.0 / 180.0 * np.pi * Robot.control_timestep
            target_change_tdy_right_slack = np.clip(
                -target_change_tdy + 20.0 / 180.0 * np.pi * Robot.control_timestep, target_change_tdy_left_slack, None
            )

            rew_change_in_abs_tdy = near_quadratic_bound(
                smoothed_change_in_tdy,
                target_change_tdy,
                target_change_tdy_left_slack,
                target_change_tdy_right_slack,
                "gaussian",
                1.6,
                0.1
            )

            # rew_change_in_abs_tdy *= max(np.abs(target_delta_yaw) / np.pi, 1/5)
            rew_change_in_abs_tdy *= 5.0
            info_dict["rew_change_in_abs_TDY"] = rew_change_in_abs_tdy
            rew_step += rew_change_in_abs_tdy
            self.rew = rew_step

    def reset_reward(self, Robot: BaseWalker, info_dict: dict[str, Any],
                     termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
                     randomState: np.random.RandomState) -> None:
        super().reset_reward(Robot, info_dict, termination_provider_triggered, randomState)
        self.queue_tdy.clear()
        info_dict["rew_change_in_abs_TDY"] = 0.0


class JoystickPolicyETHRewardProvider(JoystickPolicyRewardProvider[BaseWalker]):
    def __init__(self) -> None:
        super().__init__()
        self.rew = 0.0

    def get_reward(self) -> float:
        return self.rew

    def reset_reward(
            self,
            Robot: BaseWalker,
            info_dict: dict[str, Any],
            termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
            randomState: np.random.RandomState
    ) -> None:
        self.rew = 0.0

    def step_reward(
            self,
            Robot: BaseWalker,
            action_target_qpos: np.ndarray,
            target_goal_world_delta: np.ndarray,
            target_goal_local: np.ndarray,
            target_yaw: float,
            target_delta_yaw: float,
            target_velocity: float,
            velocity_to_goal: float,
            change_in_abs_target_delta_yaw: float,
            target_custom_data: "Any | None",
            enable_target_custom_obs: bool,
            info_dict: dict[str, Any],
            randomState: np.random.RandomState
    ) -> None:
        target_angular_velocity = 45.0 / 180.0 * np.pi * (target_delta_yaw / np.pi)

        local_velocity = Robot.get_3d_local_velocity()
        linear_velocity = Robot.get_3d_linear_velocity()
        angular_velocity = Robot.get_3d_angular_velocity()
        roll, pitch, yaw = Robot.get_roll_pitch_yaw()

        target_linear_velocity_2d = np.array([
            np.cos(yaw) * target_velocity,
            np.sin(yaw) * target_velocity,
        ])

        velocity_term = 1.0 * Robot.control_timestep * calculate_gaussian_activation(
            linear_velocity[:2] - target_linear_velocity_2d)
        angular_term = 0.5 * Robot.control_timestep * calculate_gaussian_activation(
            angular_velocity[2] - target_angular_velocity)
        linear_velocity_penalty = -4 * Robot.control_timestep * linear_velocity[2] ** 2
        angular_velocity_penalty = -0.05 * Robot.control_timestep * np.linalg.norm(angular_velocity[:2]) ** 2
        joint_motion_penalty = -0.001 * Robot.control_timestep * (
                    np.linalg.norm(Robot.get_joint_qacc()) ** 2 + np.linalg.norm(Robot.get_joint_qvel()) ** 2)
        joint_torque_penalty = -0.00002 * Robot.control_timestep * np.linalg.norm(Robot.get_joint_torques()) ** 2

        info_dict["velocity_term"] = velocity_term
        info_dict["angular_term"] = angular_term
        info_dict["linear_velocity_penalty"] = linear_velocity_penalty
        info_dict["angular_velocity_penalty"] = angular_velocity_penalty
        info_dict["joint_motion_penalty"] = joint_motion_penalty
        info_dict["joint_torque_penalty"] = joint_torque_penalty

        self.rew = 2 + velocity_term + angular_term + linear_velocity_penalty + angular_velocity_penalty + joint_motion_penalty + joint_torque_penalty
