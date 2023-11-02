import gym
import gym.spaces
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from rail_walker_interface import BaseWalker, BaseWalkerLocalizable, JoystickPolicyTargetProvider, JoystickPolicyTerminationConditionProvider
from rail_mujoco_walker import RailSimWalkerDMControl, JoystickPolicyProviderWithDMControl, JoystickPolicyDMControlTask, add_arrow_to_mjv_scene
from dm_control.mujoco.engine import Physics as EnginePhysics
from mujoco import MjvScene

from rail_walker_interface.robot.robot import BaseWalkerLocalizable

STUCK_THRESHOLD = 20 * 60 * 5
FRICTION_LIST = [0.8, 0.4, 1.2, 0.5]
# FRICTION_LIST = [0.2, 0.2, 1.5, 1.5]
CENTROIDS = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]]) / 2.0

class RouteFollow2DTraversible:
    @property
    def goal_threshold(self) -> float:
        """
        The threshold for when the goal is considered reached.
        """
        raise NotImplementedError()
    
    def fraction_of_projection_to_route(self, point : np.ndarray) -> float:
        """
        Project a point onto the route, return the scalar fraction of the projection
        """
        target_delta = self.current_goal_point - self.current_start_point
        target_delta_mag = np.linalg.norm(target_delta)
        target_delta_unit = target_delta / target_delta_mag
        current_delta = point[:2] - self.current_start_point
        current_delta_mag = np.inner(current_delta, target_delta_unit)
        return current_delta_mag / target_delta_mag

    @property
    def has_crouchable_points(self) -> bool:
        """
        Whether the route has crouchable points.
        """
        raise NotImplementedError()
    
    def should_crouch(self, robot_pos : np.ndarray) -> bool:
        """
        Whether the robot should crouch.
        """
        raise NotImplementedError()

    """
    Interface that defines a traversible route
    """
    @property
    def current_start_point(self) -> np.ndarray:
        """
        The current start point of the route.
        """
        raise NotImplementedError()
    
    @property
    def current_start_heading_range(self) -> Tuple[float,float]:
        """
        The current start heading of the route.
        """
        raise NotImplementedError()
    
    @property
    def current_goal_point(self) -> np.ndarray:
        """
        The current goal point of the route.
        """
        raise NotImplementedError()
    
    @property
    def is_complete(self) -> bool:
        """
        Whether the route is complete.
        """
        raise NotImplementedError()

    @property
    def robot_stuck(self) -> bool:
        """
        Whether the robot is stuck.
        """
        raise NotImplementedError()
    
    def get_fraction_complete(self, robot_pos : np.ndarray) -> float:
        """
        Get the percentage of the route that is complete.
        """
        raise NotImplementedError()
    
    def tick(self, robot_pos: np.ndarray, robot : BaseWalker, random_state: np.random.RandomState) -> bool:
        """
        Update the route
        """
        raise NotImplementedError()
    
    def reset(
        self,
        robot_pos: np.ndarray,
        robot : BaseWalker,
        random_state : np.random.RandomState
    ):
        """
        Reset the route
        """
        raise NotImplementedError()


class RouteFollow2DTraversibleGraph(RouteFollow2DTraversible):
    """
    A graph of goals that the car can drive to. Once the car arrives at a goal,
    the goal will be changed to one of its successors.
    """
    def __init__(
        self, 
        scale : float, 
        goals : List[Tuple[np.ndarray, Tuple[float,float]]], 
        connections = Dict[int, int], 
        start_idx : int = 0, 
        goal_threshold : float = 0.5, 
        crouchable_callbacks : List[Callable[[np.ndarray], bool]] = [],
    ):
        RouteFollow2DTraversible.__init__(self)
        self.goals = goals
        self.connections = connections
        self.start_idx = start_idx
        self._goal_threshold = goal_threshold
        self.crouchable_callbacks = crouchable_callbacks

        self.goal_reprs = []
        self.edge_reprs = []

        self.current_start_idx = 0
        self.current_goal_idx = 0

        self.scale = scale
        
        self.pos_history = []
        self.goal_generator_random_state = None
        self._variable_friction = False

    @property
    def has_crouchable_points(self) -> bool:
        """
        Whether the route has crouchable points.
        """
        return len(self.crouchable_callbacks) > 0
    
    def should_crouch(self, robot_pos : np.ndarray) -> bool:
        """
        Whether the robot should crouch.
        """
        for callback in self.crouchable_callbacks:
            if callback(robot_pos):
                return True
        return False

    def set_goal_generator_random_state(self, random_state : np.random.RandomState):
        self.goal_generator_random_state = random_state

    @property
    def goal_threshold(self) -> float:
        return self._goal_threshold
    
    @goal_threshold.setter
    def goal_threshold(self, value : float):
        self._goal_threshold = value

    @property
    def variable_friction(self) -> float:
        return self._variable_friction
    
    @variable_friction.setter
    def variable_friction(self, value : float):
        self._variable_friction = value

    @property
    def current_start_point(self) -> np.ndarray:
        return self.goals[self.current_start_idx][0] * self.scale

    @property
    def current_start_heading_range(self) -> Tuple[float,float]:
        return self.goals[self.current_start_idx][1]
    
    @property
    def current_goal_point(self):
        return self.goals[self.current_goal_idx][0] * self.scale

    @property
    def is_complete(self) -> bool:
        return self.current_goal_idx == self.current_start_idx

    @property
    def robot_stuck(self) -> bool:
        """
        Whether the robot is stuck for 5 minutes. or if it goes "out of bounds"
        """
        # timesteps = 20 * 60 * 5
        if len(self.pos_history) < STUCK_THRESHOLD:
            return False
        elif np.any(np.abs(self.pos_history[-1]) / self.scale > 0.9):
            # import ipdb; ipdb.set_trace()
            return True 
        else:
            start_pos = self.pos_history[0]
            diffs = np.array(self.pos_history) - start_pos
            norm_diffs = np.linalg.norm(diffs, axis=1)
            return np.all(norm_diffs < self.goal_threshold)
        
    def set_goal(self, goal_idx, physics = None):
        """
        Set a new goal and update the renderables to match.
        """
        for idx, repr in enumerate(self.goal_reprs):
            # opacity = 1.0 if idx == goal_idx else 0.0
            opacity = 1 if idx < 7 else 0.0
            if physics is not None:
                physics.bind(repr).rgba = (*repr.rgba[:3], opacity)
            else:
                repr.rgba = (*repr.rgba[:3], opacity)

        self.current_goal_idx = goal_idx
        self._ticks_at_current_goal = 0

    def get_fraction_complete(self, robot_pos: np.ndarray) -> float:
        return 0.0

    def tick(self, robot_pos: np.ndarray, robot : BaseWalker, random_state: np.random.RandomState):
        """
        Update the goal if the car was at the current goal for at least one tick.
        We need the delay so that the car can get the high reward for reaching
        the goal before the goal changes.
        """
        if isinstance(robot.unwrapped(), RailSimWalkerDMControl):
            physics = robot.mujoco_walker._last_physics
        else:
            physics = None

        self.pos_history.append(robot_pos[:2])
        self.pos_history = self.pos_history[-STUCK_THRESHOLD:]
        
        if np.linalg.norm(np.array(robot_pos)[:2] - self.current_goal_point) < self.goal_threshold:
            # import ipdb; ipdb.set_trace()
            came_from_idx =  self.current_start_idx
            self.current_start_idx = self.current_goal_idx
            # possible_next_goals = self.connections[self.current_start_idx]
            # disallows going back to the previous goal (and therefore having to turn 180)
            possible_next_goals = [connection for connection in self.connections[self.current_start_idx] if connection != came_from_idx]

            if self.goal_generator_random_state is not None:
                sampled_next_goal = self.goal_generator_random_state.choice(possible_next_goals)
            else:
                sampled_next_goal = random_state.choice(possible_next_goals)
            self.set_goal(sampled_next_goal, physics)

            return True
        else:
            return False

    def reset(
        self,
        robot_pos: np.ndarray,
        robot : BaseWalker,
        random_state : np.random.RandomState
    ):
        self.pos_history = []
        # self.current_start_idx = random.randint(0, len(self.goals) - 1)
        if isinstance(robot.unwrapped(), RailSimWalkerDMControl):
            physics = robot.mujoco_walker._last_physics
        else:
            physics = None
        
        self.current_start_idx = self.start_idx
        possible_goals = self.connections[self.start_idx]
        if self.goal_generator_random_state is not None:
            sampled_goal = self.goal_generator_random_state.choice(possible_goals)
        else:
            sampled_goal = random_state.choice(possible_goals)
        self.set_goal(sampled_goal, physics)
        self._ticks_at_current_goal = 0

    def add_sim_renderables(self, mjcf_root, height_lookup, show_edges=False, render_height_offset=5.0):
        """
        Add renderables to the mjcf root to visualize the goals and (optionally) edges.
        """
        self.clear_sim_renderables()

        self.goal_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.1",
                                    rgba=(0.0, 1.0, 0.0, 0.5),
                                    group=0,
                                    pos=(g[0][0] * self.scale, g[0][1] * self.scale, height_lookup((g[0][0] * self.scale, g[0][1] * self.scale)) + render_height_offset))
            for g in self.goals
        ]

        self.edge_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="cylinder",
                                    size="0.05",
                                    rgba=(1, 1, 1, 0.5),
                                    fromto=(self.goals[s][0][0] * self.scale, self.goals[s][0][1] * self.scale, height_lookup((self.goals[s][0][0] * self.scale, self.goals[s][0][1] * self.scale)) + render_height_offset,
                                            self.goals[g][0][0] * self.scale, self.goals[g][0][1] * self.scale, height_lookup((self.goals[g][0][0] * self.scale, self.goals[g][0][1] * self.scale)) + render_height_offset))
            for s in self.connections.keys() for g in self.connections[s]
            if show_edges
        ]

    def clear_sim_renderables(self):
        if self.goal_reprs is not None and len(self.goal_reprs) > 0:
            for r in self.goal_reprs:
                r.remove()
            self.goal_reprs = []
        
        if self.edge_reprs is not None and len(self.edge_reprs) > 0:
            for r in self.edge_reprs:
                r.remove()
            self.edge_reprs = []

    def __del__(self):
        self.clear_sim_renderables()

class RouteFollow2DTraversibleRoute(RouteFollow2DTraversible):
    def __init__(
        self, 
        scale : float, 
        route_points : List[Tuple[np.ndarray,Tuple[float, float]]], 
        goal_threshold : float = 0.5, 
        is_infinite : bool = False,
        crouchable_callbacks : List[Callable[[np.ndarray],bool]] = [],
    ):
        RouteFollow2DTraversible.__init__(self)
        self.route_points = route_points
        self._goal_threshold = goal_threshold
        self.scale = scale
        self.is_infinite = is_infinite
        self.crouchable_callbacks = crouchable_callbacks

        self.goal_reprs = []
        self.edge_reprs = []

        self.current_start_idx = 0

        self.pos_history = []
        self._variable_friction = False

    @property
    def has_crouchable_points(self) -> bool:
        """
        Whether the route has crouchable points.
        """
        return len(self.crouchable_callbacks) > 0
    
    def should_crouch(self, robot_pos : np.ndarray) -> bool:
        """
        Whether the robot should crouch.
        """
        for callback in self.crouchable_callbacks:
            if callback(robot_pos):
                return True
        return False

    @property
    def goal_threshold(self) -> float:
        return self._goal_threshold
    
    @goal_threshold.setter
    def goal_threshold(self, value : float):
        self._goal_threshold = value

    @property
    def variable_friction(self) -> float:
        return self._variable_friction
    
    @variable_friction.setter
    def variable_friction(self, value : float):
        self._variable_friction = value

    @property
    def current_start_point(self) -> np.ndarray:
        return self.route_points[self.current_start_idx][0] * self.scale
    
    @property
    def current_start_heading_range(self) -> Tuple[float,float]:
        return self.route_points[self.current_start_idx][1]
    
    @property
    def current_goal_point(self):
        if self.is_infinite:
            return self.route_points[(self.current_start_idx + 1) % len(self.route_points)][0] * self.scale
        else:
            assert self.current_start_idx + 1 < len(self.route_points)
            return self.route_points[self.current_start_idx + 1][0] * self.scale
    
    @property
    def is_complete(self) -> bool:
        return (not self.is_infinite) and self.current_start_idx >= len(self.route_points) - 1

    @property
    def robot_stuck(self) -> bool:
        """
        Whether the robot is stuck for 5 minutes.
        """
        # timesteps = 20 * 60 * 5
        if len(self.pos_history) < STUCK_THRESHOLD:
            return False
        elif np.any(np.abs(self.pos_history[-1]) / self.scale > 0.9):
            # import ipdb; ipdb.set_trace()
            return True 
        else:
            start_pos = self.pos_history[0]
            diffs = np.array(self.pos_history) - start_pos
            norm_diffs = np.linalg.norm(diffs, axis=1)
            return np.all(norm_diffs < self.goal_threshold)

    @is_complete.setter
    def is_complete(self, value : bool):
        if value:
            self.current_start_idx = len(self.route_points) - 1

    def get_idx_length(self, idx : int) -> float:
        return np.linalg.norm(self.route_points[(idx + 1) % len(self.route_points)][0] - self.route_points[idx % len(self.route_points)][0])
    
    def get_fraction_complete(self, robot_pos: np.ndarray) -> float:
        if self.is_infinite:
            all_idx_lengths = [self.get_idx_length(i) for i in range(len(self.route_points))]
        else:
            all_idx_lengths = [self.get_idx_length(i) for i in range(len(self.route_points) - 1)]
        
        current_route_fraction = self.fraction_of_projection_to_route(robot_pos[:2])
        current_route_fraction = np.clip(current_route_fraction, 0.0, 1.0)
        
        current_length = np.sum(all_idx_lengths[:self.current_start_idx]) + current_route_fraction * all_idx_lengths[self.current_start_idx]
        total_length = np.sum(all_idx_lengths)
        return current_length / total_length

    def tick(self, robot_pos: np.ndarray, robot: BaseWalker, random_state: np.random.RandomState):
        self.pos_history.append(robot_pos[:2])
        self.pos_history = self.pos_history[-STUCK_THRESHOLD:]

        dist_pos_to_goal = np.linalg.norm(robot_pos - self.current_goal_point)
        if dist_pos_to_goal < self.goal_threshold:
            self.current_start_idx += 1
            if self.is_infinite:
                self.current_start_idx %= len(self.route_points)
            else:
                self.current_start_idx = min(self.current_start_idx, len(self.route_points) - 1)
            return True
        else:
            return False
    
    def reset(self, robot_pos: np.ndarray, robot: BaseWalker, random_state: np.random.RandomState):
        self.current_start_idx = 0
        self.pos_history = []
    
    def add_sim_renderables(self, mjcf_root, height_lookup, show_edges=False, render_height_offset=5.0):
        """
        Add renderables to the mjcf root to visualize the goals and (optionally) edges.
        """
        self.clear_sim_renderables()

        self.goal_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.1",
                                    rgba=(0.0, 1.0, 0.0, 0.5),
                                    group=0,
                                    pos=(g[0][0] * self.scale, g[0][1] * self.scale, height_lookup((g[0][0] * self.scale, g[0][1] * self.scale)) + render_height_offset))
            for g in self.route_points
        ]

        if show_edges:
            self.edge_reprs = [
                mjcf_root.worldbody.add('site',
                                        type="cylinder",
                                        size="0.05",
                                        rgba=(1, 1, 1, 0.5),
                                        fromto=(self.route_points[g_idx][0][0] * self.scale, self.route_points[g_idx][0][1] * self.scale, height_lookup((self.route_points[g_idx][0][0] * self.scale, self.route_points[g_idx][0][1] * self.scale)) + render_height_offset,
                                                self.route_points[g_idx+1][0][0] * self.scale, self.route_points[g_idx+1][0][1] * self.scale, height_lookup((self.route_points[g_idx+1][0][0] * self.scale, self.route_points[g_idx+1][0][1] * self.scale)) + render_height_offset))
                for g_idx in range(len(self.route_points) - 1)
            ]
            if self.is_infinite:
                self.edge_reprs.append(
                    mjcf_root.worldbody.add(
                        'site',
                        type="cylinder",
                        size="0.05",
                        rgba=(1, 1, 1, 0.5),
                        fromto=(
                            self.route_points[-1][0][0] * self.scale, self.route_points[-1][0][1] * self.scale, height_lookup((self.route_points[-1][0][0] * self.scale, self.route_points[-1][0][1] * self.scale)) + render_height_offset,
                            self.route_points[0][0][0] * self.scale, self.route_points[0][0][1] * self.scale, height_lookup((self.route_points[0][0][0] * self.scale, self.route_points[0][0][1] * self.scale)) + render_height_offset
                        )
                    )
                )
        
    
    def clear_sim_renderables(self):
        if self.goal_reprs is not None and len(self.goal_reprs) > 0:
            for r in self.goal_reprs:
                r.remove()
            self.goal_reprs = []
        
        if self.edge_reprs is not None and len(self.edge_reprs) > 0:
            for r in self.edge_reprs:
                r.remove()
            self.edge_reprs = []

    def __del__(self):
        self.clear_sim_renderables()
    
def dfs_longest(points : List[Tuple[np.ndarray,...]], connections : Dict[int,List[int]], start_idx : int) -> List[int]:
    """
    Perform a depth-first search to find the longest path from the start_idx to any goal. The length of the path is defined to be the L2 distance between the points.
    """
    longest_path = []
    longest_path_length = 0.0
    def dfs_recursive(idx : int, path : List[int], path_length : float):
        nonlocal longest_path, longest_path_length
        if path_length > longest_path_length:
            longest_path = path
            longest_path_length = path_length
        for n in connections[idx]:
            if n in path:
                continue
            new_segment_length = np.linalg.norm(points[n][0] - points[idx][0])
            dfs_recursive(n, path + [n], path_length + new_segment_length)
    
    dfs_recursive(start_idx, [start_idx], 0.0)
    return longest_path

class JoystickPolicyRouteFollow2DTraversibleTerminationProvider(JoystickPolicyTerminationConditionProvider[BaseWalker]):
    def __init__(
        self,
        traversible: RouteFollow2DTraversible
    ):
        self.traversible = traversible
    
    def should_terminate(self) -> bool:
        return self.traversible.is_complete or self.traversible.robot_stuck
    
    def step_termination_condition(
        self, 
        Robot: RailSimWalkerDMControl, 
        target_goal_world_delta: np.ndarray,
        target_goal_local: np.ndarray,
        target_yaw : float,
        target_delta_yaw: float, 
        target_velocity: float,
        velocity_to_goal: float, 
        change_in_abs_target_delta_yaw : float, 
        target_custom_data: Optional[Any],
        enable_target_custom_obs : bool,
        info_dict: dict[str,Any],
        randomState: np.random.RandomState
    ) -> None:
        pass

    def reset_termination_condition(
        self, 
        Robot: RailSimWalkerDMControl, 
        info_dict: dict[str,Any], 
        termination_provider_triggered,
        randomState: np.random.RandomState
    ) -> None:
        pass

"""
This class provides a target yaw provider based on the RouteFollow2DTraversible.
"""
class JoystickPolicyRouteFollow2DTraversibleTargetProvider(JoystickPolicyTargetProvider[BaseWalkerLocalizable],JoystickPolicyProviderWithDMControl):
    def __init__(
        self,
        traversible: RouteFollow2DTraversible,
        lookahead_distance : float = 1.0,
        target_linear_velocity : float = 0.5,
    ):
        JoystickPolicyTargetProvider.__init__(self)
        JoystickPolicyProviderWithDMControl.__init__(self)
        self.traversible = traversible
        self.lookahead_distance = lookahead_distance
        self.finished_lap_count = 0
        self.times_stuck = 0
        self.target_linear_velocity = target_linear_velocity

        self._lookahead_point = np.zeros(2)
        self._lookahead_point_delta = np.zeros(2)
        self._should_crouch = False

    def get_target_custom_data(self) -> Any | None:
        return {
            "should_crouch": self._should_crouch
        }

    def get_target_custom_data_observable_spec(self) -> gym.Space | None:
        return None if not self.traversible.has_crouchable_points else gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def get_target_custom_data_observable(self) -> Any | None:
        return None if not self.traversible.has_crouchable_points else np.array([1.0 if self._should_crouch else 0.0], dtype=np.float32)
    
    def get_target_goal_world_delta(self, Robot: BaseWalkerLocalizable) -> np.ndarray:
        return self._lookahead_point_delta
    
    def get_target_velocity(self, Robot: BaseWalkerLocalizable) -> float:
        return self.target_linear_velocity
    
    def is_target_velocity_fixed(self) -> bool:
        return True

    def has_target_changed(self) -> bool:
        return True
    
    def step_target(self, Robot: BaseWalkerLocalizable, info_dict: dict[str, Any], randomState: np.random.RandomState) -> None:
        pass

    def after_step_target(
        self, 
        Robot: BaseWalkerLocalizable, 
        info_dict: dict[str,Any], 
        randomState : np.random.RandomState
    ) -> None:
        robot_pos = Robot.get_3d_location()
        current_segment_complete = self.traversible.tick(robot_pos[:2], Robot, randomState)

        if self.traversible.variable_friction:
            scaled_pos = robot_pos[:2] / self.traversible.scale
            distances = np.linalg.norm(CENTROIDS - scaled_pos, axis=1)
            closest_idx = np.argmin(distances)
            friction_val = FRICTION_LIST[closest_idx]
            if Robot.foot_friction != friction_val:
                print(f"===================== Setting friction to {friction_val} =====================")
                Robot.foot_friction = friction_val

        self._should_crouch = self.traversible.should_crouch(robot_pos)
        info_dict["should_crouch"] = self._should_crouch
        info_dict['traversible_current_lap_fraction_complete'] = self.traversible.get_fraction_complete(robot_pos[:2])
        info_dict['2d_location'] = robot_pos[:2]
        is_traversible_complete = self.traversible.is_complete or (
            isinstance(self.traversible, RouteFollow2DTraversibleRoute) and
            self.traversible.current_start_idx == 0 and
            self.traversible.is_infinite and
            current_segment_complete
        )

        if is_traversible_complete:
            self.finished_lap_count += 1

        if self.traversible.robot_stuck:
            self.times_stuck += 1
        
        if not self.traversible.is_complete:
            self.update_lookahead_point(robot_pos)
        
        info_dict["traversible_finished_lap_count"] = self.finished_lap_count
        info_dict["robot_stuck_count"] = self.times_stuck

    def update_lookahead_point(
        self,
        robot_pos: np.ndarray,
    ):
        segment_fraction = self.traversible.fraction_of_projection_to_route(robot_pos[:2])
        clipped_segment_fraction = np.clip(segment_fraction, 0.0, 1.0)
        clipped_segment_point = self.traversible.current_start_point + clipped_segment_fraction * (self.traversible.current_goal_point - self.traversible.current_start_point)
        dist_to_goal_point = np.linalg.norm(self.traversible.current_goal_point - clipped_segment_point)
        
        if dist_to_goal_point > self.lookahead_distance:
            unit_vector_to_segment_goal = self.traversible.current_goal_point - self.traversible.current_start_point
            unit_vector_to_segment_goal /= np.linalg.norm(unit_vector_to_segment_goal)
            
            self._lookahead_point = clipped_segment_point + unit_vector_to_segment_goal * self.lookahead_distance
        else:
            self._lookahead_point = self.traversible.current_goal_point
        self._lookahead_point_delta = self._lookahead_point - robot_pos[:2]
    
    def render_scene_callback(self, task : JoystickPolicyDMControlTask, physics : EnginePhysics, scene : MjvScene) -> None:
        height = task.robot.get_3d_location()[2]
        arrow_start = np.array([*self.traversible.current_start_point[:2], height + 0.3])
        arrow_end = np.array([*self.traversible.current_goal_point[:2], height + 0.3])
        add_arrow_to_mjv_scene(scene, arrow_start, arrow_end, 0.01, np.array([0.0, 0.0, 1.0, 1.0]))

    def reset_target(
        self, 
        Robot: BaseWalkerLocalizable, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        info_dict["traversible_finished_lap_count"] = self.finished_lap_count
        info_dict["robot_stuck_count"] = self.times_stuck
        
        robot_location = Robot.get_3d_location()
        if self.traversible.is_complete:
            print("Resetting Traversible")
            self.traversible.reset(robot_location[:2], Robot, randomState)

        if self.traversible.variable_friction:
            scaled_pos = robot_location[:2] / self.traversible.scale
            distances = np.linalg.norm(CENTROIDS - scaled_pos, axis=1)
            closest_idx = np.argmin(distances)
            friction_val = FRICTION_LIST[closest_idx]
            # friction_val = 0.2
            if Robot.foot_friction != friction_val:
                print(f"===================== Resetting friction to {friction_val} =====================")
                Robot.foot_friction = friction_val
        
        self._should_crouch = self.traversible.should_crouch(robot_location)
        info_dict["should_crouch"] = self._should_crouch
        info_dict['traversible_current_lap_fraction_complete'] = self.traversible.get_fraction_complete(robot_location[:2])
        info_dict['2d_location'] = robot_location[:2]
        self.update_lookahead_point(robot_location)