from copy import deepcopy
import time

import hydra.utils
import numpy as np
from panda_robot import PandaArm
import rospy
from scipy.spatial.transform.rotation import Rotation as R

from robot_io.robot_interface.base_robot_interface import BaseRobotInterface, GripperState
from robot_io.utils.utils import euler_to_quat, np_quat_to_scipy_quat, pos_orn_to_matrix

# from robot_io.control.IKfast_panda import IKfast


class PandaRosInterface(BaseRobotInterface):
    def __init__(
        self,
        force_threshold,
        torque_threshold,
        k_gains,
        d_gains,
        ik_solver,
        # rest_pose,
        workspace_limits,
        use_impedance,
    ):
        """
        :param force_threshold: list of len 6 or scalar (gets repeated for all values)
        :param torque_threshold: list of len 7 or scalar (gets repeated for all values)
        :param k_gains: joint impedance k_gains
        :param d_gains: joint impedance d_gains
        :param ik_solver: kdl or ik_fast
        :param rest_pose: joint_positions for null space (only for ik_fast)
        :param use_impedance: use joint impedance control
        """
        self.name = "panda"
        rospy.init_node("PandaRosInterface")
        self.robot = PandaArm()
        self.gripper = self.robot.get_gripper()
        self.gripper_state = GripperState.CLOSED
        self.open_gripper()
        self.gripper_state = GripperState.OPEN
        self.set_collision_threshold(force_threshold, torque_threshold)
        self.activate_controller(use_impedance, k_gains, d_gains)
        self.prev_j_des = self.robot._neutral_pose_joints
        self.ik_solver = ik_solver
        # if ik_solver == 'ik_fast':
        #     self.ik_fast = hydra.utils.instantiate(ik_cfg)
        #     self.ik_fast = IKfast(rp=rest_pose,
        #                           ll=self.robot.joint_limits()['lower'],
        #                           ul=self.robot.joint_limits()['upper'],
        #                           weights=(10, 8, 6, 6, 2, 2, 1),
        #                           num_angles=50)
        super().__init__()

    def move_to_neutral(self):
        self.robot.move_to_neutral()

    def get_state(self):
        return deepcopy(self.robot.state())

    def get_tcp_pos_orn(self):
        pos, orn = self.robot.ee_pose()
        return pos, np_quat_to_scipy_quat(orn)

    def get_tcp_pose(self):
        return pos_orn_to_matrix(*self.robot.ee_pose())

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        if len(target_orn) == 3:
            target_orn = euler_to_quat(target_orn)
        j_des = self._inverse_kinematics(target_pos, target_orn)
        print(target_pos, target_orn)
        print(j_des)
        self.robot.move_to_joint_position(j_des)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        if len(target_orn) == 3:
            target_orn = euler_to_quat(target_orn)
        j_des = self._inverse_kinematics(target_pos, target_orn)

        self.robot.set_joint_positions_velocities(j_des, [0] * 7)  # impedance control command (see documentation at )

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        current_pos, current_orn = self.get_tcp_pos_orn()
        target_pos = current_pos + rel_target_pos
        if len(rel_target_orn == 3):
            target_orn = (R.from_euler("xyz", rel_target_orn) * R.from_quat(current_orn)).as_quat()
        elif len(rel_target_orn == 4):
            target_orn = (R.from_quat(rel_target_orn) * R.from_quat(current_orn)).as_quat()
        else:
            raise ValueError
        self.move_async_cart_pos_abs_ptp(target_pos, target_orn)

    def open_gripper(self, blocking=False):
        if self.gripper_state == GripperState.CLOSED:
            self.gripper.move_joints(width=0.2, speed=3, wait_for_result=blocking)
            self.gripper_state = GripperState.OPEN

    def close_gripper(self, blocking=False):
        if self.gripper_state == GripperState.OPEN:
            self.gripper.grasp(
                width=0.02, force=5, speed=5, epsilon_inner=0.005, epsilon_outer=0.02, wait_for_result=blocking
            )
            self.gripper_state = GripperState.CLOSED

    def set_collision_threshold(self, force_threshold, torque_threshold):
        """
        :param force_threshold: list of len 6 or scalar (gets repeated for all values)
        :param torque_threshold: list of len 7 or scalar (gets repeated for all values)
        """
        if isinstance(force_threshold, (int, float)):
            force_threshold = [force_threshold] * 6  # cartesian force threshold
        else:
            assert len(force_threshold) == 6
        if isinstance(torque_threshold, (int, float)):
            torque_threshold = [torque_threshold] * 7  # joint torque threshold
        else:
            assert len(torque_threshold) == 7
        self.robot.set_collision_threshold(joint_torques=torque_threshold, cartesian_forces=force_threshold)

    def activate_controller(self, use_impedance, k_gains, d_gains):
        """
        Activate joint impedance controller.
        :param use_impedance: use joint impedance control
        :param k_gains: List of len 7 or scalar, which is interpreted as a scaling factor for default k_gains
        :param d_gains: List of len 7 or scalar, which is interpreted as a scaling factor for default k_gains
        """
        cm = self.robot.get_controller_manager()
        if not use_impedance:
            controller_name = "franka_ros_interface/position_joint_position_controller"
        else:
            controller_name = "franka_ros_interface/effort_joint_impedance_controller"
        if not cm.is_running(controller_name):
            if cm.current_controller is not None:
                cm.stop_controller(cm.current_controller)
            cm.start_controller(controller_name)
        time.sleep(1)

        if use_impedance:
            default_k_gains = np.array([1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0])
            default_d_gains = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

            if isinstance(k_gains, (float, int)):
                assert 0.2 < k_gains <= 1
                k_gains = list(default_k_gains * k_gains)
            elif k_gains is None:
                k_gains = list(default_k_gains)

            if isinstance(d_gains, (float, int)):
                assert 0.5 <= d_gains <= 1
                d_gains = list(default_d_gains * d_gains)
            elif d_gains is None:
                d_gains = list(default_d_gains)

            assert len(k_gains) == 7
            assert len(d_gains) == 7

            ctrl_cfg_client = cm.get_current_controller_config_client()
            ctrl_cfg_client.set_controller_gains(k_gains, d_gains)

    def _inverse_kinematics(self, target_pos, target_orn):
        """
        :param target_pos: cartesian target position
        :param target_orn: cartesian target orientation
        :return: status (True if solution was found), target_joint_positions
        """
        if self.ik_solver == "kdl":
            status, j = self.robot.inverse_kinematics(target_pos, target_orn)
        elif self.ik_solver == "ik_fast":
            status, j = self.ik_fast.inverse_kinematics(target_pos, target_orn)
        else:
            raise NotImplementedError

        if status:
            j_des = j
        else:
            print("Did not find IK Solution")
            j_des = self.prev_j_des
        self.prev_j_des = j_des
        return j_des
