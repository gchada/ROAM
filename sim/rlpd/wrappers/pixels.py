from typing import Any, Dict, List, Optional, Tuple

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper

from rlpd.wrappers.frame_stack import FrameStack
from rlpd.wrappers.repeat_action import RepeatAction
from rlpd.wrappers.universal_seed import UniversalSeed


class VisionObservationWrapper(PixelObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        pixels_only: bool = True,
        render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
    ):
        super().__init__(env, pixels_only, render_kwargs, pixel_keys)
        self._env = env

    def _render(self, *args, **kwargs):
        return self._env.task.depth


def wrap_pixels(
    env: gym.Env,
    action_repeat: int,
    image_size: int = 84,
    num_stack: Optional[int] = 3,
    camera_id: int = 0,
    pixel_keys: Tuple[str, ...] = ("pixels",),
) -> gym.Env:
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = UniversalSeed(env)
    env = gym.wrappers.RescaleAction(env, -1, 1)

    env = VisionObservationWrapper(
        env,
        pixels_only=False,
        render_kwargs={
            "pixels": {
                "height": 240,
                "width": 320,
                "camera_id": camera_id,
            }
        },
        pixel_keys=pixel_keys,
    )

    if num_stack is not None:
        env = FrameStack(env, num_stack=num_stack)

    env = gym.wrappers.ClipAction(env)

    return env, pixel_keys
