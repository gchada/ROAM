from ..joystick_policy.route_follow import RouteFollow2DTraversible
from rail_walker_interface import JoystickPolicyResetter, JoystickPolicyTerminationConditionProvider, BaseWalkerInSim, BaseWalker
from typing import Any, Optional, Tuple
import numpy as np
import transforms3d as tr3d

class JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(JoystickPolicyResetter[BaseWalkerInSim]):
    def __init__(self, traversible : RouteFollow2DTraversible) -> None:
        super().__init__()
        self.traversible = traversible
        self._inited = False

    def reset(
        self, 
        Robot: BaseWalker, 
        info_dict: dict[str,Any], 
        termination_provider_triggered: JoystickPolicyTerminationConditionProvider,
        randomState: np.random.RandomState
    ) -> None:
        assert isinstance(Robot.unwrapped(), BaseWalkerInSim) and isinstance(Robot.unwrapped(), BaseWalker)

        if not self._inited or self.traversible.is_complete:
            robot_pos = Robot.get_3d_location()
            self.traversible.reset(robot_pos[:2], Robot, randomState)
            self._inited = True
            
            respawn_yaw_range : Tuple[float, float] = self.traversible.current_start_heading_range
            respawn_yaw = randomState.rand() * (respawn_yaw_range[1] - respawn_yaw_range[0]) + respawn_yaw_range[0]
            
            target_quat = tr3d.euler.euler2quat(0, 0, respawn_yaw)
            Robot.reset_2d_location(self.traversible.current_start_point, target_quat)