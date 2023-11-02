import gym
from dm_control.locomotion import arenas
from .register_helper import iter_formatted_register_env
from rail_mujoco_walker import RailSimWalkerDMControl, Go1SimWalker, A1SimWalker, JoystickPolicyDMControlTask, HeightFieldArena, HEIGHTFIELD_ARENA_GOALS, CrouchingHeightfieldArena
from rail_walker_interface import JoystickPolicy, BaseWalker, WalkerVelocitySmoother
from typing import Any, Optional
from .wrappers import *
from ..joystick_policy import *
from ..joystick_policy_mujoco import *
import numpy as np
from .register_config import *
from mujoco_utils import composer_utils
from dm_control import composer
from dm_env_wrappers import ActionNoiseWrapper

def make_sim_env(
    robot: RailSimWalkerDMControl,
    joystick_policy_parameters : dict[str,Any],
    floor_override : str = "floor",
    render_height : int = 256,
    render_width : int = 256,
    height_scale : float = 5.0,
) -> gym.Env:
    print("Making sim env")
    if isinstance(robot.unwrapped(), Go1SimWalker):
        robot.control_timestep = 0.05
        robot.control_subtimestep = 0.005
    elif isinstance(robot, A1SimWalker):
        robot.control_timestep = 0.05
        robot.control_subtimestep = 0.002
    
    joystick_policy = JoystickPolicy(
        robot = robot,
        **joystick_policy_parameters
    )

    print("Joystick Policy Parameters", joystick_policy_parameters)
    
    # Initialize floor regardless of floor_override
    if floor_override == "floor":
        floor = arenas.Floor(size=(CONFIG_FLOOR_SCALE,CONFIG_FLOOR_SCALE))
        floor._top_camera.remove()
    elif floor_override == "heightfield_arena":
        floor = HeightFieldArena(CONFIG_FLOOR_SCALE, height_scale=height_scale)
    elif floor_override == "crouching_heightfield_arena":
        floor = CrouchingHeightfieldArena(CONFIG_FLOOR_SCALE)
    else:
        raise ValueError(f"Invalid floor_override {floor_override}, must be one of ['floor', 'heightfield_arena', 'crouching_heightfield_arena']")
    
    task = JoystickPolicyDMControlTask(
        joystick_policy=joystick_policy,
        floor=floor
    )

    env = composer_utils.Environment(
        task=task,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=False
    )
    # env = composer.Environment(
    #     task=task,
    #     strip_singleton_obs_buffer_dim=True,
    # )
    # env = ActionNoiseWrapper(env, scale=0.05)

    env = DMControlMultiCameraRenderWrapper(
        env,
        task.render_scene_callback,
        render_height,
        render_width
    )
    env = RailWalkerMujocoComplianceWrapper(env)
    return env

iter_formatted_register_env(
    id_format = "{}ResetMujoco-v0",
    make_entry_point = make_sim_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range = CONFIG_ACTION_RANGE
                )
            }),
            ("A1", None, lambda kwargs: {
                **kwargs,
                "robot": A1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            })
        ],
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "floor_override": "floor",
        "joystick_policy_parameters": dict(
            reward_provider=ResetRewardProvider(CONFIG_USE_ENERGY_PENALTY),
            target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
            termination_providers=[],
            truncation_providers=[],
            resetters=[ResetPolicyResetter(init_dist=np.array([1.0, 0.0, 0.0]))],
            target_observable = None,
            enabled_observables = ['joints_pos', 'imu']
        )
    }
)

iter_formatted_register_env(
    id_format = "{}EasyResetMujoco-v0",
    make_entry_point = make_sim_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            }),
            ("A1", None, lambda kwargs: {
                **kwargs,
                "robot": A1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            })
        ],
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "floor_override": "floor",
        "joystick_policy_parameters": dict(
            reward_provider=ResetRewardProvider(CONFIG_USE_ENERGY_PENALTY),
            target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
            termination_providers=[],
            truncation_providers=[],
            resetters=[ResetPolicyResetter(init_dist=np.array([0.5, 0.5, 0.0]))],
            target_observable = None,
            enabled_observables = ['joints_pos', 'imu']
        )
    }
)

iter_formatted_register_env(
    id_format = "{}GatedResetMujoco-v0",
    make_entry_point = make_sim_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            }),
            ("A1", None, lambda kwargs: {
                **kwargs,
                "robot": A1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            })
        ],
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "floor_override": "floor",
        "joystick_policy_parameters": dict(
            reward_provider=GatedResetRewardProvider(CONFIG_USE_ENERGY_PENALTY),
            target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
            termination_providers=[],
            truncation_providers=[],
            resetters=[ResetPolicyResetter()],
            target_observable = None,
            enabled_observables = ['joints_pos', 'imu']
        )
    }
)

iter_formatted_register_env(
    id_format = "{}GatedEasyResetMujoco-v0",
    make_entry_point = make_sim_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            }),
            ("A1", None, lambda kwargs: {
                **kwargs,
                "robot": A1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            })
        ],
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        "floor_override": "floor",
        "joystick_policy_parameters": dict(
            reward_provider=GatedResetRewardProvider(CONFIG_USE_ENERGY_PENALTY),
            target_yaw_provider=JoystickPolicyForwardOnlyTargetProvider(),
            termination_providers=[],
            truncation_providers=[],
            resetters=[ResetPolicyResetter(init_dist=np.array([0.5, 0.5, 0.0]))],
            target_observable = None,
            enabled_observables = ['joints_pos', 'imu']
        )
    }
)

def make_sanity_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyForwardOnlyTargetProvider()
    kwargs['joystick_policy_parameters']['resetters'] = [JoystickPolicyPointInSimResetter(np.zeros(2),0)]
    return kwargs

def make_full_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['floor_override'] = "heightfield_arena"
    to_follow = HEIGHTFIELD_ARENA_GOALS["full_dense"]
    idx_route = dfs_longest(to_follow["points"], to_follow["connections"], 0)
    route_to_follow = [to_follow["points"][idx] for idx in idx_route]
    print("Route to follow:", route_to_follow)
    traversible = RouteFollow2DTraversibleRoute(
        CONFIG_FLOOR_SCALE,
        route_to_follow,
        is_infinite=idx_route[0] in to_follow["connections"][idx_route[-1]]
    )
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollow2DTraversibleTargetProvider(traversible, lookahead_distance=1)
    kwargs['joystick_policy_parameters']['truncation_providers'].append(JoystickPolicyRouteFollow2DTraversibleTerminationProvider(traversible))
    kwargs['joystick_policy_parameters']['resetters'] = [
        JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False), 
        JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(traversible) # For aligning the robot with the route when initializing or route is resetting
    ]
    return kwargs

def make_squished_full_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_full_goal_route_lambda_sim(kwargs)
    kwargs['height_scale'] = 10.0
    return kwargs

def make_unsquished_full_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_full_goal_route_lambda_sim(kwargs)
    kwargs['height_scale'] = 2.5
    return kwargs

def make_goal_graph_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['floor_override'] = "heightfield_arena"
    to_follow = HEIGHTFIELD_ARENA_GOALS["dense_inner"]
    idx_route = dfs_longest(to_follow["points"], to_follow["connections"], 0)
    route_to_follow = [to_follow["points"][idx] for idx in idx_route]
    print("Route to follow:", route_to_follow)
    traversible = RouteFollow2DTraversibleRoute(
        CONFIG_FLOOR_SCALE,
        route_to_follow,
        is_infinite=idx_route[0] in to_follow["connections"][idx_route[-1]]
    )
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollow2DTraversibleTargetProvider(traversible, lookahead_distance=1)
    kwargs['joystick_policy_parameters']['truncation_providers'].append(JoystickPolicyRouteFollow2DTraversibleTerminationProvider(traversible))
    kwargs['joystick_policy_parameters']['resetters'] = [
        JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False), 
        JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(traversible) # For aligning the robot with the route when initializing or route is resetting
    ]
    return kwargs

def make_goal_graph_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['floor_override'] = "heightfield_arena"
    to_follow = HEIGHTFIELD_ARENA_GOALS["dense_inner"]
    traversible = RouteFollow2DTraversibleGraph(
        scale = CONFIG_FLOOR_SCALE,
        goals = to_follow['points'],
        connections= to_follow['connections'],

    )
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollow2DTraversibleTargetProvider(traversible, lookahead_distance=1)
    kwargs['joystick_policy_parameters']['truncation_providers'].append(JoystickPolicyRouteFollow2DTraversibleTerminationProvider(traversible))
    kwargs['joystick_policy_parameters']['resetters'] = [
        JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False), 
        JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(traversible) # For aligning the robot with the route when initializing or route is resetting
    ]
    return kwargs

def make_unsquished_goal_graph_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_goal_graph_lambda_sim(kwargs)
    kwargs['height_scale'] = 2.5
    return kwargs

def make_squished_goal_graph_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_goal_graph_lambda_sim(kwargs)
    kwargs['height_scale'] = 10.0
    return kwargs

def make_full_goal_graph_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['floor_override'] = "heightfield_arena"
    to_follow = HEIGHTFIELD_ARENA_GOALS["full_dense"]
    traversible = RouteFollow2DTraversibleGraph(
        scale = CONFIG_FLOOR_SCALE,
        goals = to_follow['points'],
        connections= to_follow['connections'],

    )
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollow2DTraversibleTargetProvider(traversible, lookahead_distance=1)
    kwargs['joystick_policy_parameters']['truncation_providers'].append(JoystickPolicyRouteFollow2DTraversibleTerminationProvider(traversible))
    kwargs['joystick_policy_parameters']['resetters'] = [
        JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False), 
        JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(traversible) # For aligning the robot with the route when initializing or route is resetting
    ]
    return kwargs

def make_squished_full_goal_graph_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_full_goal_graph_lambda_sim(kwargs)
    kwargs['height_scale'] = 10.0
    return kwargs

def make_obstacle_full_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_full_goal_route_lambda_sim(kwargs)
    kwargs['floor_override'] = "crouching_heightfield_arena"
    return kwargs

def make_crouching_full_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs = make_full_goal_route_lambda_sim(kwargs)
    kwargs['joystick_policy_parameters']['enable_target_custom_obs'] = False
    traversible : RouteFollow2DTraversibleRoute = kwargs['joystick_policy_parameters']['target_yaw_provider'].traversible
    kwargs['floor_override'] = "crouching_heightfield_arena"
    traversible.crouchable_callbacks.append(
        CrouchingHeightfieldArena.get_should_crouch_callback(CONFIG_FLOOR_SCALE)
    )
    return kwargs

def make_hiking_goal_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['floor_override'] = "heightfield_arena"
    to_follow = HEIGHTFIELD_ARENA_GOALS["hike"]
    idx_route = dfs_longest(to_follow["points"], to_follow["connections"], 0)
    route_to_follow = [to_follow["points"][idx] for idx in idx_route]
    print("Route to follow:", route_to_follow)
    traversible = RouteFollow2DTraversibleRoute(
        CONFIG_FLOOR_SCALE,
        route_to_follow,
        is_infinite=idx_route[0] in to_follow["connections"][idx_route[-1]]
    )
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollow2DTraversibleTargetProvider(traversible, lookahead_distance=1)
    kwargs['joystick_policy_parameters']['truncation_providers'].append(JoystickPolicyRouteFollow2DTraversibleTerminationProvider(traversible))
    kwargs['joystick_policy_parameters']['resetters'] = [
        JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False), 
        JoystickPolicyRouteFollow2DTraversibleSimRouteResetter(traversible) # For aligning the robot with the route when initializing or route is resetting
    ]
    return kwargs

def make_circular_route_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    # goal_graph = GoalGraph.from_circle(np.zeros(2), 10.0, True, 8, 1.0)
    # goal_route = GoalRoute(goal_graph, start_idx=0)
    # kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyRouteFollowTargetProvider(goal_route, reset_robot_after_route_reset=False)
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyCircleFollowTargetProvider(10.0, 1.0)
    # kwargs['joystick_policy_parameters']['termination_providers'].append(JoystickPolicyRouteFollowTerminationProvider(goal_route))
    kwargs['joystick_policy_parameters']['resetters'] = [JoystickPolicyLastPositionAndYawResetter(random_respawn_yaw=False, position_if_init=np.array([10.0, 0.0]))]
    return kwargs

def make_autojoystick_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyAutoJoystickTargetProvider()
    kwargs['joystick_policy_parameters']['resetters'] = [JoystickPolicyPointInSimResetter(np.zeros(2),0)]
    kwargs['robot'] = WalkerVelocitySmoother(
        kwargs['robot'], 
        np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
    )
    return kwargs

def make_autojoystick_simple_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyAutoJoystickSimpleTargetProvider()
    kwargs['joystick_policy_parameters']['resetters'] = [JoystickPolicyPointInSimResetter(np.zeros(2),0)]
    kwargs['robot'] = WalkerVelocitySmoother(
        kwargs['robot'], 
        np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
    )
    return kwargs

def make_autojoystick_simple_plus_lambda_sim(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyAutoJoystickSimplePlusTargetProvider()
    kwargs['joystick_policy_parameters']['resetters'] = [JoystickPolicyPointInSimResetter(np.zeros(2),0)]
    kwargs['robot'] = WalkerVelocitySmoother(
        kwargs['robot'], 
        np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
    )
    return kwargs

def make_smoothed_target_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicySmoothedTargetProvider(
        kwargs['joystick_policy_parameters']['target_yaw_provider'],
        max_rate=5.0/180.0*np.pi,
    )
    return kwargs

def make_limited_target_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_yaw_provider'] = JoystickPolicyLimitedTargetProvider(
        kwargs['joystick_policy_parameters']['target_yaw_provider'],
        max_delta_angle=30.0/180.0*np.pi,
    )
    return kwargs


def make_target_delta_yaw_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyTargetDeltaYawObservable()
    return kwargs

def make_cos_sin_target_delta_yaw_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyCosSinTargetDeltaYawObservable()
    return kwargs

def make_target_delta_yaw_and_dist_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = JoystickPolicyTargetDeltaYawAndDistanceObservable()
    return kwargs

def make_empty_obs_lambda(
    kwargs: dict[str,Any]
) -> dict[str,Any]:
    kwargs['joystick_policy_parameters']['target_observable'] = None
    return kwargs

def make_strict_reward_lambda(
    kwargs: dict[str,Any]
) -> dict[str, Any]:
    kwargs['joystick_policy_parameters']['reward_provider'] = JoystickPolicyStrictRewardProvider()
    return kwargs

def make_sep_reward_lambda(
    kwargs: dict[str,Any]
) -> dict[str, Any]:
    kwargs['joystick_policy_parameters']['reward_provider'] = JoystickPolicySeperateRewardProvider()
    return kwargs

def make_eth_reward_lambda(
    kwargs: dict[str, Any]
) -> dict[str, Any]:
    kwargs["joystick_policy_parameters"]["reward_provider"] = JoystickPolicyETHRewardProvider()
    return kwargs

def make_awi_reward_lambda(
    kwargs: dict[str, Any]
) -> dict[str, Any]:
    kwargs["joystick_policy_parameters"]["reward_provider"] = WalkInTheParkRewardProvider()
    return kwargs

iter_formatted_register_env(
    id_format = "{}{}Mujoco-{}{}-{}-v0",
    make_entry_point = make_sim_env,
    format_args_list = [
        [
            ("Go1", None, lambda kwargs: {
                **kwargs,
                "robot": Go1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE,
                )
            }),
            ("A1", None, lambda kwargs: {
                **kwargs,
                "robot": A1SimWalker(
                    Kp = CONFIG_KP,
                    Kd = CONFIG_KD,
                    action_interpolation = CONFIG_ACTION_INTERPOLATION,
                    limit_action_range=CONFIG_ACTION_RANGE
                )
            }),
        ],
        [
            ("GoalRoute1", None, make_full_goal_route_lambda_sim),
            ("GoalRoute1Obstacle", None, make_obstacle_full_goal_route_lambda_sim),
            ("GoalRoute1Crouchable", None, make_crouching_full_goal_route_lambda_sim),
            ("GoalRoute2", None, make_hiking_goal_route_lambda_sim),
            ("GoalGraph", None, make_goal_graph_lambda_sim),
            ("GoalGraphStretched", None, make_unsquished_goal_graph_lambda_sim),
            ("GoalGraphSquished", None, make_squished_goal_graph_lambda_sim),
            ("GoalGraphRoute", None, make_goal_graph_route_lambda_sim),
            ("GoalGraphFull", None, make_full_goal_graph_lambda_sim),
            ("GoalGraphFullSquished", None, make_squished_full_goal_graph_lambda_sim),
            ("GoalGraphFullSquishedRoute", None, make_squished_full_goal_route_lambda_sim),
            # ("GoalRoute3", None, make_smallinnergraphreversed_goal_route_lambda_sim),
            # ("GoalRoute4", None, make_edges_goal_route_lambda_sim),
            # ("GoalRoute5", None, make_densering_goal_route_lambda_sim),
            # ("GoalRoute6", None, make_smallinnerring_goal_route_lambda_sim),
            ("AutoJoystick", None, make_autojoystick_lambda_sim),
            ("AutoJoystickSimple", None, make_autojoystick_simple_lambda_sim),
            ("AutoJoystickSimplePlus", None, make_autojoystick_simple_plus_lambda_sim),
            ("CircularRoute", None, make_circular_route_lambda_sim),
            ("Sanity", None, make_sanity_lambda_sim),
        ],
        [
            ("Smoothed", None, make_smoothed_target_lambda),
            ("Limited", None, make_limited_target_lambda),
            ("", None, lambda kwargs: kwargs)
        ],
        [
            ("TDY", None, make_target_delta_yaw_obs_lambda),
            ("CosSinTDY", None, make_cos_sin_target_delta_yaw_obs_lambda),
            ("TDYDist", None, make_target_delta_yaw_and_dist_obs_lambda),
            ("Empty", None, make_empty_obs_lambda)
        ],
        [
            ("StrictRew", None, make_strict_reward_lambda),
            ("SepRew", None, make_sep_reward_lambda),
            ("ETHRew", None, make_eth_reward_lambda),
            ("AWIRew", None, make_awi_reward_lambda)
        ]
    ],
    base_register_kwargs = dict(
        
    ),
    base_make_env_kwargs = dict(
        
    ),
    base_make_env_kwargs_callback = lambda kwargs: {
        **kwargs,
        # "floor_override": "floor",
        "joystick_policy_parameters": dict(
            termination_providers=[JoystickPolicyRollPitchTerminationConditionProvider(30/180*np.pi, -25/180*np.pi, 20/180*np.pi)],
            truncation_providers=[],
            enabled_observables=[
                "joints_pos",
                "joints_vel",
                "imu",
                "sensors_local_velocimeter",
                "torques",
                "foot_forces_normalized",
            ],
        )
    }
)
