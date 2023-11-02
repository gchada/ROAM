from collections.abc import Callable, Iterable, Mapping
import numpy as np
import robot_interface
import time
from typing import Any
from serde import serde, SerdeError
from serde.msgpack import from_msgpack, to_msgpack
from dataclasses import dataclass
import math
import errno
import socket
import multiprocessing
from ctypes import c_float, c_bool
import struct
import select
import os
import psutil

def empty_obs():
    return Go1RemoteObservation(
        angular_velocity=np.zeros(3),
        framequat_wijk=np.zeros(4),
        roll_pitch_yaw=np.zeros(3),
        acceleration_local=np.zeros(3),
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        joint_torques=np.zeros(12),
        foot_contacts=np.zeros(4),
        left_joystick=np.zeros(2),
        right_joystick=np.zeros(2),
        motor_temperatures=np.zeros(12)
    )

@serde
@dataclass
class Go1RemoteObservation:
    angular_velocity: np.ndarray
    framequat_wijk: np.ndarray
    roll_pitch_yaw: np.ndarray
    acceleration_local: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torques: np.ndarray
    foot_contacts: np.ndarray
    left_joystick : np.ndarray
    right_joystick : np.ndarray
    motor_temperatures : np.ndarray

@serde
@dataclass
class Go1RemoteConfigMsg:
    action_interpolation : bool
    Kp : float
    Kd : float
    ppf : float
    control_timestep : float


@serde
@dataclass
class Go1RemoteActionMsg:
    target_action : np.ndarray

LOW_LEVEL_CONTROL = 0xff
UDP_SERVER_IP_BASIC = "192.168.123.10"
UDP_SERVER_PORT_BASIC = 8007
UDP_CLIENT_PORT = 8080
POS_STOP_F = math.pow(10,9)
VEL_STOP_F = 16000.0
MAX_TORQUE = np.array([23.7, 23.7, 35.55] * 4)
REAL_CONTROL_TIMESTEP = 0.005
HIBERNATION_QPOS = np.array([0.0 / 180 * np.pi, 80.0/180*np.pi, -150.0 / 180 * np.pi] * 4)

class DataPack:
    def __init__(self, deal_command_callback) -> None:
        self.remaining_data = b""
        self.deal_command_callback = deal_command_callback
    
    def feed_data(self, dat: bytes, custom : Any) -> Any:
        self.remaining_data += dat
        return self.stream_decode(custom)
    
    def clear_data(self) -> None:
        self.remaining_data = b""

    def stream_decode(self, custom : Any) -> Any:
        decoded = True
        ret = None
        while decoded:
            decoded, ret = self.try_decode(custom)
        return ret

    def try_decode(self, custom: Any) -> tuple[bool, Any]:
        if len(self.remaining_data) < 3:
            return False, None
        command = self.remaining_data[0:1]
        length = int.from_bytes(self.remaining_data[1:3], 'big')

        if len(self.remaining_data) < length + 3:
            return False, None
        
        if len(self.remaining_data) < length + 3 + 3:
            last_data = True
        else:
            next_length = int.from_bytes(self.remaining_data[length+3+1:length+3+3], 'big')
            last_data = not len(self.remaining_data) >= (length+3)+3+next_length

        data = self.remaining_data[3:length + 3]
        ret = self.deal_command_callback(command, data, last_data, custom)
        self.remaining_data = self.remaining_data[length + 3:]
        return True, ret
    
    @staticmethod
    def encode(command : bytes, data : bytes) -> bytes:
        length = len(data)
        enc = command + length.to_bytes(2, 'big') + data
        return enc


class Go1RemoteRunnerControlThread(multiprocessing.Process):
    def __init__(
        self, 
        seconds_to_hibernate : float,
        Kp : float,
        Kd : float,
        action_interpolation : bool,
        control_timestep : float,
        ppf : float,
    ):
        super().__init__(target=self.run, daemon=False)
        self.pipe_comm, self._internal_pipe = multiprocessing.Pipe(duplex=True)
        
        # Value that shouldn't be changed after initialization
        self._seconds_to_hibernate = seconds_to_hibernate
        
        # Control thread owned values
        self._udp = None
        self._cmd_to_send = None
        self._last_received_low_state = None
        self._last_joint_pos = None
        self._last_joint_vel = None
        self._safety = None
        
        self._last_action = None
        self._current_action = None
        self._current_action_time = time.time()
        self._last_command_time = time.time()
        self._last_substep_time = time.time()
        self._real_max_torque = MAX_TORQUE * ppf
        self._hibernate = True
        
        # Synchronized Values
        self.action_interpolation = multiprocessing.Value(c_bool, action_interpolation)
        self.Kp = multiprocessing.Value(c_float, Kp)
        self.Kd = multiprocessing.Value(c_float, Kd)
        self.ppf = multiprocessing.Value(c_float, ppf)
        self.control_timestep = multiprocessing.Value(c_float,control_timestep)

    def run(self) -> None:
        # Raise the priority of the control thread
        pid = os.getpid()
        # process = psutil.Process(pid)
        # process.nice(0)

        print(f"Control Thread Started (PID = {pid})")
        t = time.time()
        self._udp = robot_interface.UDP(LOW_LEVEL_CONTROL, UDP_CLIENT_PORT, UDP_SERVER_IP_BASIC, UDP_SERVER_PORT_BASIC)
        self._safety = robot_interface.Safety(robot_interface.LeggedType.Go1)
        self._cmd_to_send = robot_interface.LowCmd()
        self._last_received_low_state = robot_interface.LowState()
        self._udp.InitCmdData(self._cmd_to_send)
        self._last_joint_pos = np.zeros(12)
        self._last_joint_vel = np.zeros(12)
        self._last_action = None
        self._current_action = None
        self._current_action_time = t
        self._last_command_time = t
        self._last_substep_time = 0
        self._hibernate = True
        
        self._real_max_torque = MAX_TORQUE * self.ppf.value
        control_timestep = self.control_timestep.value
        Kp = self.Kp.value
        Kd = self.Kd.value
        action_interpolation = self.action_interpolation.value

        try:
            while True:
                t = time.time()
                dt = t - self._last_substep_time
                if dt < REAL_CONTROL_TIMESTEP:
                    time_to_sleep = REAL_CONTROL_TIMESTEP - dt
                    time.sleep(time_to_sleep)
                    t += time_to_sleep
                    dt = REAL_CONTROL_TIMESTEP
                self._last_substep_time = t
                
                if not self._hibernate:
                    if (
                    t - self._last_command_time > self._seconds_to_hibernate
                    ):
                        self.hibernate()
                    self.substep(t, action_interpolation, control_timestep, Kp, Kd)
                else:
                    time.sleep(0.1)
                
                if self._internal_pipe.poll():
                    msg = self._internal_pipe.recv()
                    if isinstance(msg, Go1RemoteActionMsg):
                        self._last_action = self._current_action
                        self._current_action = msg.target_action
                        self._current_action_time = t
                        self._last_command_time = t
                        if self._hibernate:
                            self.cancel_hibernate(self._current_action)
                    elif msg == "o":
                        # self._last_command_time = t
                        # if self._hibernate:
                        #     self.cancel_hibernate()
                        self._internal_pipe.send(self.generate_obs())
                    elif msg == "c":
                        ppf = self.ppf.value
                        self._real_max_torque = MAX_TORQUE * ppf
                        control_timestep = self.control_timestep.value
                        Kp = self.Kp.value
                        Kd = self.Kd.value
                        print("New config applied!")
                        print("Kp: ", Kp, ", Kd:", Kd, ", control_timestep: ", control_timestep, ", action_interpolation: ", action_interpolation, ", ppf: ", ppf)
                    else:
                        continue
        finally:
            print(f"Control Thread Exited (PID={pid})")

    def hibernate(self):
        if self._hibernate:
            return
        print("Starting to hibernate")
        # Slowly interpolate to the hibernation position
        last_t = 0
        start_joint_pos = self._last_joint_pos
        delta_action = HIBERNATION_QPOS - start_joint_pos
        inter_steps = 400
        control_timestep = self.control_timestep.value
        action_interpolation = self.action_interpolation.value
        Kp = self.Kp.value
        Kd = self.Kd.value

        for i in range(inter_steps):
            t = time.time()
            dt = t - last_t
            if dt < REAL_CONTROL_TIMESTEP:
                t_to_sleep = REAL_CONTROL_TIMESTEP - dt
                time.sleep(t_to_sleep)
                t += t_to_sleep
                dt = REAL_CONTROL_TIMESTEP
            last_t = t
            target_action = start_joint_pos + delta_action * (i + 1) / inter_steps
            self.substep(t, action_interpolation, control_timestep, Kp, Kd, target_action)
        
        pid = os.getpid()
        process = psutil.Process(pid)
        process.nice(20)
        self._hibernate = True
    
    def cancel_hibernate(self, new_target_action : np.ndarray | None = None):
        if not self._hibernate:
            return

        print("Cancelling Hibernation")
        pid = os.getpid()
        process = psutil.Process(pid)
        process.nice(-20)
        # Read in the last received state
        self.set_init_comm_cmd()
        self._udp.SetSend(self._cmd_to_send)
        self._udp.Send()
        time.sleep(REAL_CONTROL_TIMESTEP)
        self._udp.Recv()
        self._udp.GetRecv(self._last_received_low_state)

        last_joint_pos = np.zeros(12)
        last_joint_vel = np.zeros(12)
        for i in range(12):
            motor_state_i = self._last_received_low_state.motorState[i]
            last_joint_pos[i] = motor_state_i.q
            last_joint_vel[i] = motor_state_i.dq
        
        self._last_joint_pos = last_joint_pos
        self._last_joint_vel = last_joint_vel
        # self._last_joint_pos = np.array([self._last_received_low_state.motorState[i].q for i in range(12)])
        # self._last_joint_vel = np.array([self._last_received_low_state.motorState[i].dq for i in range(12)])

        # Set hibernate to false
        self._hibernate = False

        # Quickly interpolate to the new target action
        if new_target_action is not None:
            control_timestep = self.control_timestep.value
            Kp = self.Kp.value
            Kd = self.Kd.value
            action_interpolation = self.action_interpolation.value

            last_t = 0
            start_joint_pos = self._last_joint_pos
            delta_action = new_target_action - start_joint_pos
            inter_steps = 150
            for i in range(inter_steps):
                t = time.time()
                dt = t - last_t
                if dt < REAL_CONTROL_TIMESTEP:
                    t_to_sleep = REAL_CONTROL_TIMESTEP - dt
                    time.sleep(t_to_sleep)
                    t += t_to_sleep
                    dt = REAL_CONTROL_TIMESTEP
                last_t = t
                target_action = start_joint_pos + delta_action * (i + 1) / inter_steps
                self.substep(t, action_interpolation, control_timestep, Kp, Kd, target_action)
            return last_t
    
    @property
    def is_hibernating(self) -> bool:
        return self._hibernate

    def substep(self, t : float, action_interpolation : bool, control_timestep : float, Kp : float, Kd : float, action_override : np.ndarray | None = None) -> None:
        if self._hibernate:
            return
        self._udp.Recv()
        self._udp.GetRecv(self._last_received_low_state)
        # for i in range(12):
        #     motor_state_i = self._last_received_low_state.motorState[i]
        #     self._last_joint_pos[i] = motor_state_i.q
        #     self._last_joint_vel[i] = motor_state_i.dq
        self._last_joint_pos = np.array([self._last_received_low_state.motorState[i].q for i in range(12)])
        self._last_joint_vel = np.array([self._last_received_low_state.motorState[i].dq for i in range(12)])

        real_action = self._current_action if action_override is None else action_override
        if real_action is None:
            self.set_init_comm_cmd()
            self._udp.SetSend(self._cmd_to_send)
            self._udp.Send()
            return
        if action_interpolation and self._last_action is not None:
            dt = t - self._current_action_time
            ai_progress = dt / control_timestep
            ai_progress = max(min(ai_progress, 1.0), 0.0)
            real_action = self._last_action + (real_action - self._last_action) * ai_progress
        
        clipped_action = self.cutoff_action(
            real_action,
            self._last_joint_pos,
            self._last_joint_vel,
            self._real_max_torque,
            Kp,
            Kd,
        )
        self.set_action_command(real_action, Kp, Kd)
        self._safety.PositionLimit(self._cmd_to_send)
        self._safety.PowerProtect(self._cmd_to_send, self._last_received_low_state, 10)
        self._udp.SetSend(self._cmd_to_send)
        self._udp.Send()

    def generate_obs(self) -> Go1RemoteObservation:
        # wireless_remote = bytes(self._last_received_low_state.wirelessRemote) 
        wireless_remote = np.asarray(self._last_received_low_state.wirelessRemote, dtype=np.uint8)
        float_values = wireless_remote[4:4 + 4*5].view(np.float32)
        lx, rx, ry, L2, ly = float_values
        # lx = struct.unpack("f", wireless_remote[4:8])
        # rx = struct.unpack("f", wireless_remote[8:12])
        # ry = struct.unpack("f",wireless_remote[12:16])
        # ly = struct.unpack("f", wireless_remote[20:24])

        joint_torques = np.zeros(12)
        joint_temperatures = np.zeros(12,dtype = np.int8)
        for i in range(12):
            state = self._last_received_low_state.motorState[i]
            joint_torques[i] = state.tauEst
            joint_temperatures[i] = state.temperature

        return Go1RemoteObservation(
            angular_velocity=np.asarray(self._last_received_low_state.imu.gyroscope),
            framequat_wijk=np.asarray(self._last_received_low_state.imu.quaternion),
            roll_pitch_yaw=np.asarray(self._last_received_low_state.imu.rpy),
            acceleration_local=np.asarray(self._last_received_low_state.imu.accelerometer),
            joint_pos=self._last_joint_pos,
            joint_vel=self._last_joint_vel,
            joint_torques=joint_torques,
            foot_contacts=np.asarray(self._last_received_low_state.footForce),
            left_joystick=np.array([lx,ly]),
            right_joystick=np.array([rx,ry]),
            motor_temperatures=joint_temperatures,
        )

    def set_init_comm_cmd(self):
        for i in range(12):
            self._cmd_to_send.motorCmd[i].q = POS_STOP_F
            self._cmd_to_send.motorCmd[i].dq = VEL_STOP_F
            self._cmd_to_send.motorCmd[i].Kp = 0.0
            self._cmd_to_send.motorCmd[i].Kd = 0.0
            self._cmd_to_send.motorCmd[i].tau = 0.0
    
    @staticmethod
    def cutoff_action(
        current_action : np.ndarray, 
        current_qpos : np.ndarray, 
        current_dqpos : np.ndarray,
        max_torque : np.ndarray,
        Kp : float,
        Kd : float,
    ) -> np.ndarray:
        # Cutoff action to prevent robot from exerting too much torque on joints
        delta_qpos = current_action - current_qpos
        current_d_torque = Kd * current_dqpos
        # current_d_torque = np.clip(current_d_torque, -max_torque, max_torque)
        min_delta_qpos = (current_d_torque - max_torque) / Kp
        min_delta_qpos = np.minimum(min_delta_qpos, 0) # Clip so that 0 is always a valid value
        
        max_delta_qpos = (current_d_torque + max_torque) / Kp
        max_delta_qpos = np.maximum(max_delta_qpos, 0) # Clip so that 0 is always a valid value

        clipped_delta_qpos = np.clip(delta_qpos, min_delta_qpos, max_delta_qpos)
        clipped_action = current_qpos + clipped_delta_qpos
        return clipped_action

    def set_action_command(self, real_action : np.ndarray, Kp : float, Kd : float):
        real_action = real_action.astype(np.float32)
        for i in range(12):
            self._cmd_to_send.motorCmd[i].q = real_action[i]
            self._cmd_to_send.motorCmd[i].dq = 0.0
            self._cmd_to_send.motorCmd[i].Kp = Kp
            self._cmd_to_send.motorCmd[i].Kd = Kd
            self._cmd_to_send.motorCmd[i].tau = 0.0
    
    def receive_observation(self) -> Go1RemoteObservation:
        self.pipe_comm.send("o")
        return self.pipe_comm.recv()
    
    def apply_control(self, action : np.ndarray) -> None:
        self.pipe_comm.send(Go1RemoteActionMsg(action))
    
    def apply_config(
        self,
        Kp : float,
        Kd : float,
        ppf : float,
        control_timestep : float,
        action_interpolation : bool
    ):
        self.Kp.value = Kp
        self.Kd.value = Kd
        self.ppf.value = ppf
        self.control_timestep.value = control_timestep
        self.action_interpolation.value = action_interpolation
        self.pipe_comm.send("c")

class Go1RemoteRunner:
    def __init__(
        self,
        bind_address : tuple[str, int] = ("", 6001),
        seconds_to_hibernate : float = 20.0,
        read_obs_timestep : float = 0.025,
    ):
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.bind(bind_address)
        self.tcp_server.listen(1)
        self.tcp_server.setblocking(False)
        self.tcp_clients : list[socket.socket] = []
        self._bind_address = bind_address
        
        self.control_thread = Go1RemoteRunnerControlThread(
            seconds_to_hibernate,
            40.0,
            5.0,
            True,
            0.05,
            1.0
        )
        self.control_thread.start()
        self.read_obs_timestep = read_obs_timestep
        self._last_obs : Go1RemoteObservation = empty_obs()
        self._last_obs_t = 0.0

        self.data_pack = DataPack(self.deal_with_data)
    
    def send_data(self, data : bytes, target_socket : socket.socket) -> None:
        try:
            num_sent = target_socket.send(data)
        except:
            self.disconnect(target_socket)
            return
        if num_sent <= 0:
            self.disconnect(target_socket)
    
    def disconnect(self, socket : socket.socket) -> None:
        print("Disconnecting Socket", socket)
        try:
            self.tcp_clients.remove(socket)
        except ValueError:
            pass
        socket.close()
        if len(self.tcp_clients) <= 0:
            self.data_pack.clear_data()
    
    def broadcast_data(self, data : bytes) -> None:
        for client in self.tcp_clients:
            self.send_data(data, client)
    
    def deal_with_data(self, command : bytes, data : bytes, is_last_data : bool, custom : socket.socket) -> Any:
        if command == b"a":
            try:
                action_msg = from_msgpack(Go1RemoteActionMsg,data)
                if action_msg.target_action.shape == (12,):
                    target_action = action_msg.target_action.astype(np.float32)
                    self.send_data(DataPack.encode(b"p", b""), custom)
                    self.control_thread.apply_control(target_action)
                    
                    # When applying control let's send back a obs too
                    # obs_msg = to_msgpack(self._last_obs)
                    # to_send = DataPack.encode(b"o", obs_msg)
                    # self.send_data(to_send, custom)
            except SerdeError as e:
                print("Decode error", e)
            except:
                print("Error unpacking data")
        elif command == b"c":
            try:
                config_msg = from_msgpack(Go1RemoteConfigMsg,data)
                self.send_data(DataPack.encode(b"p", b""), custom)
                self.control_thread.apply_config(
                    config_msg.Kp,
                    config_msg.Kd,
                    config_msg.ppf,
                    config_msg.control_timestep,
                    config_msg.action_interpolation
                )
                self.read_obs_timestep = config_msg.control_timestep / 2

            except SerdeError as e:
                print("Decode error", e)
            except:
                print("Error unpacking data")

    @property
    def bind_address(self) -> tuple[str, int]:
        return self._bind_address
    
    def try_read_obs(self):
        t = time.time()
        if t - self._last_obs_t < self.read_obs_timestep:
            return
        elif len(self.tcp_clients) <= 0:
            return
        self._last_obs_t = t
        self._last_obs = self.control_thread.receive_observation()
        # Broadcast every time we receive a new obs
        obs_msg = to_msgpack(self._last_obs)
        to_send = DataPack.encode(b"o", obs_msg)
        self.broadcast_data(to_send)

    def run(self):
        while True:
            self.try_read_obs()
            if len(self.tcp_clients) <= 0:
                time.sleep(0.1)
            
            readable, writable, errored = select.select([self.tcp_server] + self.tcp_clients, [], [], 0.0)
            for s in readable:
                if s is self.tcp_server:
                    client_socket, address = self.tcp_server.accept()
                    self.tcp_clients.append(client_socket)
                    print ("Connection from", address)
                else:
                    try:
                        data = s.recv(1024)
                    except socket.error as e:
                        if e.errno == errno.ECONNRESET:
                            self.disconnect(s)
                            break
                        elif e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                            raise e
                        continue
                    if data:
                        self.data_pack.feed_data(data, s)
                    else:
                        self.disconnect(s)
                        break