import numpy as np
from rail_mujoco_walker.robots.sim_robot import RailSimWalkerDMControl
from rail_walker_interface import JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider
from rail_mujoco_walker import RailSimWalkerDMControl, JoystickPolicyProviderWithDMControl, JoystickPolicyDMControlTask, add_arrow_to_mjv_scene
from dm_control.mujoco.engine import Physics as EnginePhysics
import mujoco
from typing import Any
import math


"""
This class provides a Circle Follow target yaw provider.
"""
class JoystickPolicyCircleFollowTargetProvider(JoystickPolicyTargetProvider[RailSimWalkerDMControl],JoystickPolicyProviderWithDMControl):
    def __init__(
        self,
        circle_radius = 10.0,
        lookahead_distance : float = 1.0,
        target_linear_velocity : float = 0.5,
    ):
        JoystickPolicyTargetProvider.__init__(self)
        JoystickPolicyProviderWithDMControl.__init__(self)
        self.circle_radius = circle_radius
        self.lookahead_distance = lookahead_distance
        self._lookahead_point = np.zeros(2)
        self._lookahead_point_delta = np.zeros(2)
        self.target_linear_velocity = target_linear_velocity
    
    def get_target_goal_world_delta(self, Robot: RailSimWalkerDMControl) -> np.ndarray:
        return self._lookahead_point_delta
    
    def get_target_velocity(self, Robot: RailSimWalkerDMControl) -> float:
        return self.target_linear_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return True

    def step_target(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        self.update_lookahead_point(Robot.get_3d_location())

    def update_lookahead_point(
        self,
        robot_pos: np.ndarray,
    ):
        if np.allclose(robot_pos[:2], np.zeros(2)):
            self._lookahead_point = np.array([1.0,0.0]) * self.circle_radius
        else:
            angle_current = math.atan2(robot_pos[1], robot_pos[0])
            angle_lookahead = angle_current + self.lookahead_distance / self.circle_radius
            self._lookahead_point = np.array([math.cos(angle_lookahead), math.sin(angle_lookahead)]) * self.circle_radius
        self._lookahead_point_delta = self._lookahead_point - robot_pos[:2]
    
    def reset_target(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        self.update_lookahead_point(Robot.get_3d_location())

    def render_scene_callback(self, task : JoystickPolicyDMControlTask, physics : EnginePhysics, scene : mujoco.MjvScene) -> None:
        angle_lookahead = math.atan2(self._lookahead_point[1], self._lookahead_point[0])
        angle_before_lookahead = angle_lookahead - self.lookahead_distance / self.circle_radius
        arrow_start = np.concatenate([np.array([math.cos(angle_before_lookahead), math.sin(angle_before_lookahead)]) * self.circle_radius, [0.3]])
        arrow_end = np.concatenate([self._lookahead_point, [0.3]])
        add_arrow_to_mjv_scene(scene, arrow_start, arrow_end, 0.01, np.array([0.0, 0.0, 1.0, 1.0]))
