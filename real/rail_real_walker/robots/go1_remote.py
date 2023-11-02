from .go1_remote_runner import Go1RemoteActionMsg, Go1RemoteObservation, Go1RemoteConfigMsg, empty_obs, DataPack, REAL_CONTROL_TIMESTEP
import numpy as np
import time
import socket
from typing import Optional
from rail_walker_interface import BaseWalker, BaseWalkerWithFootContact, BaseWalkerWithJoystick, BaseWalkerWithJointTemperatureSensor, Walker3DVelocityEstimator
from serde.msgpack import from_msgpack, to_msgpack
import msgpack.exceptions
from serde import SerdeError
from functools import cached_property
import unitree_go1_wrapper.go1_constants as go1_constants
import transforms3d as tr3d
import errno

class Go1RealWalkerRemote(BaseWalker[Go1RemoteObservation], BaseWalkerWithFootContact, BaseWalkerWithJoystick, BaseWalkerWithJointTemperatureSensor):
    def __init__(
        self, 
        velocity_estimator: Walker3DVelocityEstimator,
        power_protect_factor : float = 0.5,
        foot_contact_threshold: np.ndarray = np.array([80, 170, 170, 170]),
        action_interpolation: bool = True,
        name: Optional[str] = "robot", 
        Kp: float = 40, 
        Kd: float = 5,
        control_timestep : float = 0.05,
        force_real_control_timestep : bool = True,
        limit_action_range : float = 1.0,
        server_addr = ("192.168.123.161",6001)
    ):  
        assert control_timestep >= 0.02
        assert power_protect_factor > 0.0 and power_protect_factor <= 1.0
        # Init communication with the robot
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_client.setblocking(True)
        self.socket_client.settimeout(0.5)
        self.socket_client.connect(server_addr)
        self.socket_client.setblocking(False)
        
        self._server_addr = server_addr
        self.data_pack = DataPack(self.deal_with_data)

        # Init config properties
        self._power_protect_factor = power_protect_factor
        self._action_interpolation = action_interpolation
        self._Kp = Kp
        self._Kd = Kd
        self._control_timestep = control_timestep

        BaseWalker.__init__(self,name, Kp, Kd, force_real_control_timestep, limit_action_range, power_protect_factor)
        BaseWalkerWithFootContact.__init__(self)
        BaseWalkerWithJoystick.__init__(self)
        BaseWalkerWithJointTemperatureSensor.__init__(self)
        
        self.velocity_estimator = velocity_estimator
        self._last_state : Optional[Go1RemoteObservation] = empty_obs()
        self._last_velocity : np.ndarray = np.zeros(3)
        self._last_velocity_estimator_t = time.time()
        self.foot_contact_threshold = foot_contact_threshold
        self.velocity_estimator.reset(self, self.get_framequat_wijk())
        self._received_pong = False
        self._should_step_velocity_estimator = False

        self._should_update_config_next = False
        self.update_config()
    
    @property
    def server_addr(self):
        return self._server_addr

    def deal_with_data(self, command : bytes, data: bytes, is_last_data : bool, custom):
        t : float = custom
        if command == b"o":
            try:
                new_state = from_msgpack(Go1RemoteObservation, data)
            except SerdeError as e:
                print("Error in remote Go1 Robot SerdeError", e)
                new_state = None
            except msgpack.exceptions.ExtraData as e:
                print("Error in remote Go1 Robot ExtraData", e)
                new_state = None
                
            if new_state is not None:
                self._last_state = new_state
            
            if self._should_step_velocity_estimator:
                # Step velocity estimator
                self.velocity_estimator.step(
                    self,
                    self.get_framequat_wijk(),
                    t - self._last_velocity_estimator_t
                )
                self._last_velocity_estimator_t = t
                self._last_velocity = self.velocity_estimator.get_3d_linear_velocity()
                self._should_step_velocity_estimator = False
        elif command == b"p":
            self._received_pong = True
        else:
            print("Unknown command", command)
        
    def try_receive_data(self):
        while True:
            try:
                data = self.socket_client.recv(8192)
                if not data:
                    raise RuntimeError("No data received from remote Go1 Runner! The TCP connection might be broken!")
                self.data_pack.feed_data(data, time.time())
                if len(data) < 8192:
                    return
            except socket.error as e:
                if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                    return
                else:
                    raise e
    
    def send_data(self, data : bytes):
        try:
            num_sent = self.socket_client.send(data)
        except:
            raise
        if num_sent <= 0:
            raise RuntimeError("No data sent to remote Go1 Runner! The TCP connection might be broken!")

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()
    
    @property
    def is_real_robot(self) -> bool:
        return True

    def update_config(self) -> None:
        to_send = self.data_pack.encode(b"c",to_msgpack(Go1RemoteConfigMsg(
            action_interpolation=self._action_interpolation,
            Kp=self._Kp,
            Kd=self._Kd,
            ppf=self._power_protect_factor,
            control_timestep=self._control_timestep
        )))
        
        self._received_pong = False
        self.send_data(to_send)
        while not self._received_pong:
            self.try_receive_data()
            time.sleep(0.01)
        self._should_update_config_next = False

    @property
    def control_timestep(self) -> float:
        return self._control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value: float) -> None:
        assert value >= 0.02
        self._control_timestep = value
        self._should_update_config_next = True

    @property
    def control_subtimestep(self) -> float:
        return REAL_CONTROL_TIMESTEP
    
    @property
    def power_protect_factor(self) -> float:
        return self._power_protect_factor
    
    @power_protect_factor.setter
    def power_protect_factor(self, value: float) -> None:
        assert value > 0.0 and value <= 1.0
        self._power_protect_factor = value
        self._should_update_config_next = True

    @property
    def Kp(self) -> float:
        return self._Kp
    
    @Kp.setter
    def Kp(self, value: float) -> None:
        self._Kp = value
        self._should_update_config_next = True
    
    @property
    def Kd(self) -> float:
        return self._Kd
    
    @Kd.setter
    def Kd(self, value: float) -> None:
        self._Kd = value
        self._should_update_config_next = True

    @property
    def action_interpolation(self) -> bool:
        return self._action_interpolation
    
    @action_interpolation.setter
    def action_interpolation(self, value: bool) -> None:
        self._action_interpolation = value
        self._should_update_config_next = True

    def receive_observation(self) -> bool:
        self._should_step_velocity_estimator = True
        # Step robot interface
        self.try_receive_data()
        return True

    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        shift_delta = np.array([-0.1, 0.0, -0.1])
        real_shift = shift_delta.repeat(4).reshape((3,4)).T.flatten()
        real_shift[3::6] = -real_shift[3::6]
        return np.array([go1_constants.GO1_HIP_INIT, go1_constants.GO1_THIGH_INIT, go1_constants.GO1_CALF_INIT] * 4) + real_shift

    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return np.array([0.0 / 180 * np.pi, 70.0/180*np.pi, -150.0 / 180 * np.pi] * 4)

    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return np.array([go1_constants.GO1_HIP_MIN, go1_constants.GO1_THIGH_MIN, go1_constants.GO1_CALF_MIN] * 4)

    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return np.array([go1_constants.GO1_HIP_MAX, go1_constants.GO1_THIGH_MAX, go1_constants.GO1_CALF_MAX] * 4)

    def reset(self) -> None:
        t = time.time()
        self._last_velocity = np.zeros(3)
        self._last_velocity_estimator_t = t
        self.velocity_estimator.reset(self, self.get_framequat_wijk())
        self._should_step_velocity_estimator = False
        self.try_receive_data()
        super().reset()

    def get_3d_linear_velocity(self) -> np.ndarray:
        return self._last_velocity
    
    def get_3d_local_velocity(self) -> np.ndarray:
        try:
            if hasattr(self.velocity_estimator,"get_3d_local_velocity"):
                return self.velocity_estimator.get_3d_local_velocity()
            else:
                return tr3d.quaternions.rotate_vector(
                    self.get_3d_linear_velocity(), 
                    tr3d.quaternions.qinverse(self.get_framequat_wijk())
                )
        except:
            print("Error in get_3d_local_velocity, continuing anyway")
            import traceback
            traceback.print_exc()
            return np.zeros(3)

    def get_3d_angular_velocity(self) -> np.ndarray:
        return self._last_state.angular_velocity
    
    def get_framequat_wijk(self) -> np.ndarray:
        return self._last_state.framequat_wijk
    
    def get_roll_pitch_yaw(self) -> np.ndarray:
        return self._last_state.roll_pitch_yaw

    def get_last_observation(self) -> Optional[Go1RemoteObservation]:
        return self._last_state

    def get_3d_acceleration_local(self) -> np.ndarray:
        return self._last_state.acceleration_local

    def get_joint_qpos(self) -> np.ndarray:
        return self._last_state.joint_pos

    def get_joint_qvel(self) -> np.ndarray:
        return self._last_state.joint_vel

    def get_joint_torques(self) -> np.ndarray:
        return self._last_state.joint_torques
    
    def get_joint_temperature_celsius(self) -> np.ndarray:
        return self._last_state.motor_temperatures.astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> bool:
        if self._should_update_config_next:
            self.update_config()
        
        to_send = self.data_pack.encode(b"a",to_msgpack(Go1RemoteActionMsg(action)))

        self.send_data(to_send)
        self.try_receive_data()
        return True
    
    def get_foot_contact(self) -> np.ndarray:
        return self.get_foot_force() >= self.foot_contact_threshold
    
    def get_foot_force(self) -> np.ndarray:
        return self._last_state.foot_contacts
    
    def get_joystick_values(self) -> np.ndarray:
        return [
            self._last_state.left_joystick,
            self._last_state.right_joystick
        ]
    
    def close(self) -> None:
        self.velocity_estimator.close()