from typing import Dict, Optional, Tuple, Callable, Any, List

import dm_control.utils.transformations as tr
import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.utils import rewards
from dm_control.mjcf.physics import Physics
from dm_control.mujoco.engine import Physics as EnginePhysics
from mujoco import MjvScene
# from dm_control.utils import transformations
from dm_control.locomotion import arenas
from dm_control import composer
from dm_control.locomotion.walkers import base
from dm_env import specs
from ..robots.sim_robot import RailSimWalkerDMControl
from rail_walker_interface import JoystickPolicy, BaseWalker, BaseWalkerWithFootContact
from functools import cached_property
from ..utils import add_arrow_to_mjv_scene, add_sphere_to_mjv_scene
from collections import OrderedDict

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.001


class JoystickPolicyProviderWithDMControl:
    def render_scene_callback(self, task: "JoystickPolicyDMControlTask", physics: EnginePhysics,
                              scene: MjvScene) -> None:
        pass


class JoystickPolicyDMControlTask(composer.Task):
    def __init__(
            self,
            joystick_policy: JoystickPolicy,
            floor: composer.Arena,
            enabled_cameras: List[str] = ["top_camera", "side_camera"]
    ):
        assert isinstance(joystick_policy.robot.unwrapped(), RailSimWalkerDMControl)
        self.joystick_policy = joystick_policy

        self._floor = floor
        # self._floor.mjcf_model.size.nconmax = 400
        # self._floor.mjcf_model.size.njmax = 2000
        self._floor.add_free_entity(self.robot.mujoco_walker)

        self.set_timesteps(joystick_policy.control_timestep, joystick_policy.control_subtimestep)

        # Add cameras
        if "side_camera" in enabled_cameras:
            self.robot.mujoco_walker.mjcf_model.worldbody.add(
                'camera',
                name='side_camera',
                pos=[0, -1, 0.5],
                xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                mode="trackcom",
                fovy=60
            )

        if "top_camera" in enabled_cameras:
            self.robot.mujoco_walker.mjcf_model.worldbody.add(
                'camera',
                name='top_camera',
                pos=[0, 0, 3],
                xyaxes=[1, 0, 0, 0, 1, 0],
                mode="trackcom",
                fovy=60.0
            )

        for obs in self.robot.mujoco_walker.observables._observables.keys():
            getattr(self.robot.mujoco_walker.observables, obs).enabled = True

        self._arrow_root = None
        self._arrow_target_yaw = None
        self._arrow_velocity = None
        self._arrow_walker_heading = None
        self._sphere_goal = None

        self.step_count = 0

    # def reload_joystick_policy(self):
    # enabled_observables = self.joystick_policy.enabled_observables
    # for obs in self.robot.mujoco_walker.observables._observables.keys():
    #     if obs not in enabled_observables:
    #         getattr(self.robot.mujoco_walker.observables, obs).enabled = False
    #     else:
    #         getattr(self.robot.mujoco_walker.observables, obs).enabled = True

    # if hasattr(self, "_target_obs") and self._target_obs is not None:
    #     self._target_obs.enabled = False
    #     self._target_obs = None

    # if self.joystick_policy.target_observable is not None:
    #     self._target_obs = observable.Generic(lambda physics: self.joystick_policy.target_observable.get_observation())
    #     self._target_obs.enabled = True
    # else:
    #     self._target_obs = None

    @property
    def root_entity(self):
        return self._floor

    @property
    def robot(self) -> RailSimWalkerDMControl:
        return self.joystick_policy.robot

    @property
    def observables(self):
        obs = super().observables
        # Enforce Observable Order
        return OrderedDict(sorted(obs.items(), key=lambda t: t[0]))

    @property
    def task_observables(self):
        task_observables = super().task_observables

        # if self._target_obs is not None:
        #     self._target_obs.enabled = True
        #     task_observables['target_obs'] = self._target_obs

        return task_observables

    @property
    def control_timestep(self) -> float:
        """Returns the agent's control timestep for this task (in seconds)."""
        return super().control_timestep

    @control_timestep.setter
    def control_timestep(self, value: float):
        """Sets the agent's control timestep for this task (in seconds)."""
        self.joystick_policy.control_timestep = value
        super().control_timestep = value

    @property
    def physics_timestep(self) -> float:
        """Returns the physics timestep for this task (in seconds)."""
        return super().physics_timestep

    @physics_timestep.setter
    def physics_timestep(self, value: float):
        """Sets the physics timestep for this task (in seconds)."""
        self.joystick_policy.control_subtimestep = value
        super().physics_timestep = value

    def set_timesteps(self, control_timestep: float, physics_timestep: float):
        self.joystick_policy.control_timestep = control_timestep
        self.joystick_policy.control_subtimestep = physics_timestep
        super().set_timesteps(control_timestep, physics_timestep)

    def get_reward(self, physics):
        rew = self.joystick_policy.get_reward()
        self.get_reward_final()
        return rew

    def get_reward_final(self):
        self.reward_final = self.joystick_policy.get_reward_final()

    def initialize_episode(self, physics: Physics, random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        self.robot.mujoco_walker._last_physics = physics
        if self.joystick_policy.reward_provider.modification == "FR_0":
            physics.named.model.jnt_stiffness[1] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FR_1":
            physics.named.model.jnt_stiffness[2] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FR_2":
            physics.named.model.jnt_stiffness[3] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FL_0":
            physics.named.model.jnt_stiffness[4] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FL_1":
            physics.named.model.jnt_stiffness[5] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FL_2":
            physics.named.model.jnt_stiffness[6] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RR_0":
            physics.named.model.jnt_stiffness[7] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RR_1":
            physics.named.model.jnt_stiffness[8] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RR_2":
            physics.named.model.jnt_stiffness[9] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RL_0":
            physics.named.model.jnt_stiffness[10] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RL_1":
            physics.named.model.jnt_stiffness[11] = 10.0
        elif self.joystick_policy.reward_provider.modification == "RL_2":
            physics.named.model.jnt_stiffness[12] = 10.0
        elif self.joystick_policy.reward_provider.modification == "FR_friction":
            physics.named.model.geom_friction['robot/FR', 0] = 0.4
        elif self.joystick_policy.reward_provider.modification == "FL_friction":
            physics.named.model.geom_friction['robot/FL', 0] = 0.4
        elif self.joystick_policy.reward_provider.modification == "RR_friction":
            physics.named.model.geom_friction['robot/RR', 0] = 0.4
        elif self.joystick_policy.reward_provider.modification == "RL_friction":
            physics.named.model.geom_friction['robot/RL', 0] = 0.4
        self._floor.initialize_episode(physics, random_state)
        self.joystick_policy.reset(random_state)

    def before_step(self, physics: Physics, action: np.ndarray, random_state: np.random.RandomState):
        self.robot.mujoco_walker._last_physics = physics

        self.step_count += 1

        if self.joystick_policy.reward_provider.modification == "friction":
            if self.step_count % 100 < 50:
                physics.named.model.geom_friction['robot/FR', 0] = 0.8
                physics.named.model.geom_friction['robot/FL', 0] = 0.8
                physics.named.model.geom_friction['robot/RR', 0] = 0.1
                physics.named.model.geom_friction['robot/RL', 0] = 0.8
            else:
                physics.named.model.geom_friction['robot/FR', 0] = 0.8
                physics.named.model.geom_friction['robot/FL', 0] = 0.01
                physics.named.model.geom_friction['robot/RR', 0] = 0.1
                physics.named.model.geom_friction['robot/RL', 0] = 0.8

        elif self.joystick_policy.reward_provider.modification == "stiffness":
            if self.step_count % 300 < 100:
                physics.named.model.jnt_stiffness[1] = 0
                physics.named.model.jnt_stiffness[2] = 0
                physics.named.model.jnt_stiffness[3] = 0
                physics.named.model.jnt_stiffness[4] = 0
                physics.named.model.jnt_stiffness[5] = 0
                physics.named.model.jnt_stiffness[6] = 0
                physics.named.model.jnt_stiffness[7] = 0
                physics.named.model.jnt_stiffness[8] = 15.0
                physics.named.model.jnt_stiffness[9] = 0

            elif 100 <= self.step_count % 300 < 200:
                physics.named.model.jnt_stiffness[1] = 0
                physics.named.model.jnt_stiffness[2] = 0
                physics.named.model.jnt_stiffness[3] = 0
                physics.named.model.jnt_stiffness[4] = 0
                physics.named.model.jnt_stiffness[5] = 15.0
                physics.named.model.jnt_stiffness[6] = 0
                physics.named.model.jnt_stiffness[7] = 0
                physics.named.model.jnt_stiffness[8] = 0
                physics.named.model.jnt_stiffness[9] = 0

            else:
                physics.named.model.jnt_stiffness[1] = 0
                physics.named.model.jnt_stiffness[2] = 15.0
                physics.named.model.jnt_stiffness[3] = 0
                physics.named.model.jnt_stiffness[4] = 0
                physics.named.model.jnt_stiffness[5] = 0
                physics.named.model.jnt_stiffness[6] = 0
                physics.named.model.jnt_stiffness[7] = 0
                physics.named.model.jnt_stiffness[8] = 0
                physics.named.model.jnt_stiffness[9] = 0

        self.joystick_policy.before_step(action, random_state)

    def before_substep(self, physics, action, random_state):
        self.robot.mujoco_walker._last_physics = physics

    def action_spec(self, physics):
        # return specs.BoundedArray(
        #     shape=(len(self.robot.joint_qpos_mins),),
        #     dtype=np.float32,
        #     minimum=self.robot.action_qpos_mins,
        #     maximum=self.robot.action_qpos_maxs
        #     #name='\t'.join([actuator.name for actuator in self.actuators])
        # )
        return self.robot.mujoco_walker.action_spec(physics)

    def after_step(self, physics, random_state):
        self.robot.mujoco_walker._last_physics = physics
        self.joystick_policy.after_step(random_state)
        # Update render arrows
        self._arrow_root = self.robot.get_3d_location()
        self._arrow_walker_heading = tr.quat_rotate(
            self.robot.get_framequat_wijk(),
            np.array([1.0, 0.0, 0.0])
        ).astype(np.float32)
        self._arrow_velocity = self.robot.get_3d_linear_velocity().astype(np.float32)
        self._arrow_target_yaw = np.array([*self.joystick_policy.target_goal_world_delta_unit, 0], dtype=np.float32)
        if hasattr(self.joystick_policy.reward_provider, "target_linear_velocity"):
            self._arrow_target_yaw *= self.joystick_policy.reward_provider.target_linear_velocity

        self._sphere_goal = np.array([*self.joystick_policy.target_goal_world_delta, 1.0]) + self._arrow_root

    def should_terminate_episode(self, physics):
        self.robot.mujoco_walker._last_physics = physics
        return False

    def get_discount(self, physics):
        return 1.0

    @cached_property
    def render_scene_callback(self):
        def render_cb(physics: EnginePhysics, scene: MjvScene):
            if isinstance(self.joystick_policy.target_yaw_provider, JoystickPolicyProviderWithDMControl):
                # Call the render_scene_callback of the target_yaw_provider
                self.joystick_policy.target_yaw_provider.render_scene_callback(self, physics, scene)

            if isinstance(self.joystick_policy.reward_provider, JoystickPolicyProviderWithDMControl):
                # Call the render_scene_callback of the reward_provider
                self.joystick_policy.reward_provider.render_scene_callback(self, physics, scene)

            for termination_provider in self.joystick_policy.termination_providers:
                if isinstance(termination_provider, JoystickPolicyProviderWithDMControl):
                    # Call the render_scene_callback of the termination_provider
                    termination_provider.render_scene_callback(self, physics, scene)

            for reset_provider in self.joystick_policy.resetters:
                if isinstance(reset_provider, JoystickPolicyProviderWithDMControl):
                    # Call the render_scene_callback of the reset_provider
                    reset_provider.render_scene_callback(self, physics, scene)

            if self._arrow_root is not None:
                # Add velocity arrow => greenish yellow
                add_arrow_to_mjv_scene(scene, self._arrow_root, self._arrow_root + self._arrow_velocity, radius=0.01,
                                       rgba=np.array([0.5, 1.0, 0.0, 1.0], dtype=np.float32))
                # Add target yaw arrow => yellow
                target_height_offset = np.array([0.0, 0.0, 0.25], dtype=np.float32)
                add_arrow_to_mjv_scene(scene, self._arrow_root + target_height_offset,
                                       self._arrow_root + target_height_offset + self._arrow_target_yaw, radius=0.01,
                                       rgba=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
                # Add target location sphere => yellow
                add_sphere_to_mjv_scene(scene, self._sphere_goal, radius=0.2,
                                        rgba=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
                # Add walker heading arrow => green
                add_arrow_to_mjv_scene(scene, self._arrow_root, self._arrow_root + self._arrow_walker_heading,
                                       radius=0.01, rgba=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))

            if isinstance(self.robot.unwrapped(), BaseWalkerWithFootContact):
                force_max_length = 0.3
                foot_forces = self.robot.get_foot_force()
                foot_contacts = self.robot.get_foot_contact()
                max_force = np.max(np.abs(foot_forces))
                foot_positions = self.robot.mujoco_walker.get_foot_positions(self.robot.mujoco_walker._last_physics)
                for i, foot_pos in enumerate(foot_positions):
                    if foot_contacts[i]:
                        foot_force = foot_forces[i]
                        # Add foot force arrow => red
                        add_arrow_to_mjv_scene(
                            scene,
                            foot_pos,
                            foot_pos + np.array([0.0, 0.0, foot_force / max_force * force_max_length]),
                            radius=0.01,
                            rgba=np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)  # blue
                        )

        return render_cb
