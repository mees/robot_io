import logging
import time

from _frankx import NetworkException
import cv2
from frankx import (
    Affine,
    ImpedanceMotion,
    JointMotion,
    JointWaypointMotion,
    LinearMotion,
    LinearRelativeMotion,
    PathMotion,
    Robot,
    StopMotion,
    Waypoint,
    WaypointMotion,
)
from frankx.gripper import Gripper
import hydra.utils
import numpy as np

from robot_io.control.rel_action_control import RelActionControl
from robot_io.robot_interface.base_robot_interface import BaseRobotInterface
from robot_io.utils.frankx_utils import to_affine
from robot_io.utils.utils import get_git_root, pos_orn_to_matrix, ReferenceType

log = logging.getLogger(__name__)

# FrankX needs continuous Euler angles around TCP, as the trajectory generation works in the Euler space.
# internally, FrankX expects orientations with the z-axis facing up, but to be consistent with other
# robot interfaces we transform the TCP orientation such that the z-axis faces down.
NE_T_EE = EE_T_NE = Affine(0, 0, 0, 0, 0, np.pi)

# align the output of force torque reading with the EE frame
WRENCH_FRAME_CONV = np.diag([-1, 1, 1, -1, 1, 1])  # np.eye(6)


class PandaFrankXInterface(BaseRobotInterface):
    """
    Robot control interface for Franka Emika Panda robot to be used on top of this Frankx fork
    (https://github.com/lukashermann/frankx)

    Args:
        fci_ip: IPv4 address of Franka Control Interface (FCI).
        urdf_path: URDF of panda robot (change default config e.g. when mounting different fingers).
        neutral_pose: Joint angles in rad.
        ll: Lower joint limits in rad.
        ul: Upper joint limits in rad.
        ik: Config of the inverse kinematic solver.
        workspace_limits: Workspace limits defined as a bounding box or as hollow cylinder.
        libfranka_params: DictConfig of params for libfranka.
        use_impedance: If True, use impedance control whenever it is possible.
        frankx_params: DictConfig of general params for Frankx.
        impedance_params: DictConfig of params for Frankx impedance motion.
        rel_action_params: DictConfig of params for relative action control.
        gripper_params: DictConfig of params for Frankx gripper.
    """

    def __init__(
        self,
        fci_ip,
        urdf_path,
        neutral_pose,
        ll,
        ul,
        ik,
        workspace_limits,
        libfranka_params,
        use_impedance,
        frankx_params,
        impedance_params,
        rel_action_params,
        gripper_params,
    ):
        self.name = "panda"
        self.neutral_pose = neutral_pose
        self.ll = ll
        self.ul = ul

        # robot
        self.robot = Robot(fci_ip, urdf_path=(get_git_root(__file__) / urdf_path).as_posix())
        self.robot.recover_from_errors()
        self.robot.set_default_behavior()
        self.libfranka_params = libfranka_params
        self.set_robot_params(libfranka_params, frankx_params)

        # impedance
        self.use_impedance = use_impedance
        self.impedance_params = impedance_params

        self.rel_action_converter = RelActionControl(
            ll=ll, ul=ul, workspace_limits=workspace_limits, **rel_action_params
        )

        self.motion_thread = None
        self.current_motion = None

        self.gripper = Gripper(fci_ip, **gripper_params)
        self.open_gripper(blocking=True)

        # F_T_NE is the transformation from nominal end-effector (NE) frame to flange (F) frame.
        F_T_NE = np.array(self.robot.read_once().F_T_NE).reshape((4, 4)).T
        self.ik_solver = hydra.utils.instantiate(ik, F_T_NE=F_T_NE)

        self.reference_type = ReferenceType.ABSOLUTE
        super().__init__(ll=ll, ul=ul)

    def __del__(self):
        self.abort_motion()

    def set_robot_params(self, libfranka_params, frankx_params):
        # params of libfranka
        self.robot.set_collision_behavior(
            libfranka_params.contact_torque_threshold,
            libfranka_params.collision_torque_threshold,
            libfranka_params.contact_force_threshold,
            libfranka_params.collision_force_threshold,
        )
        self.robot.set_joint_impedance(libfranka_params.franka_joint_impedance)

        # params of frankx
        self.robot.velocity_rel = frankx_params.velocity_rel
        self.robot.acceleration_rel = frankx_params.acceleration_rel
        self.robot.jerk_rel = frankx_params.jerk_rel

    def move_to_neutral(self):
        return self.move_joint_pos(self.neutral_pose)

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        # if self.use_impedance:
        #     log.warning("Impedance motion is not available for synchronous motions. Not using impedance.")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        return self.move_joint_pos(q_desired)

    def move_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(
            rel_target_pos, rel_target_orn, self.get_state(), self.reference_type
        )
        self.reference_type = ReferenceType.RELATIVE
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.abort_motion()
        self.robot.move(JointMotion(q_desired))

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(
            rel_target_pos, rel_target_orn, self.get_state(), self.reference_type
        )
        self.reference_type = ReferenceType.RELATIVE
        self._frankx_async_impedance_motion(target_pos, target_orn)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            log.warning("Impedance motion for cartesian PTP is currently not implemented. Not using impedance.")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.move_async_joint_pos(q_desired)

    def move_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            log.warning("Impedance motion for cartesian LIN is currently not implemented. Not using impedance.")
        self.abort_motion()
        target_pose = to_affine(target_pos, target_orn) * NE_T_EE
        self.current_motion = WaypointMotion([Waypoint(target_pose)])
        self.robot.move(self.current_motion)

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            self._frankx_async_impedance_motion(target_pos, target_orn)
        else:
            self._frankx_async_lin_motion(target_pos, target_orn)

    def move_async_joint_pos(self, joint_positions):
        if self._is_active(JointWaypointMotion):
            self.current_motion.set_next_target(joint_positions)
        else:
            if self.current_motion is not None:
                self.abort_motion()
            self.current_motion = JointWaypointMotion([joint_positions], return_when_finished=False)
            self.motion_thread = self.robot.move_async(self.current_motion)

    def move_joint_pos(self, joint_positions):
        self.reference_type = ReferenceType.JOINT
        self.abort_motion()
        success = self.robot.move(JointMotion(joint_positions))
        if not success:
            self.robot.recover_from_errors()
        return success

    def abort_motion(self):
        if self.current_motion is not None:
            self.current_motion.stop()
            self.current_motion = None
        if self.motion_thread is not None:
            #     # self.motion_thread.stop()
            self.motion_thread.join()
            self.motion_thread = None
        while 1:
            try:
                self.robot.recover_from_errors()
                break
            except NetworkException:
                time.sleep(0.01)
                continue

    def get_state(self):
        if self.current_motion is None:
            _state = self.robot.read_once()
        else:
            _state = self.current_motion.get_robot_state()
        pos, orn = self.get_tcp_pos_orn()

        state = {
            "tcp_pos": pos,
            "tcp_orn": orn,
            "joint_positions": np.array(_state.q),
            "gripper_opening_width": self.gripper.width(),
            "force_torque": WRENCH_FRAME_CONV @ np.array(_state.K_F_ext_hat_K),
            "contact": np.array(_state.cartesian_contact),
        }
        return state

    def get_tcp_pos_orn(self):
        if self.current_motion is None:
            pose = self.robot.current_pose() * EE_T_NE
        else:
            pose = self.current_motion.current_pose() * EE_T_NE
            while np.all(pose.translation() == 0):
                pose = self.current_motion.current_pose() * EE_T_NE
                time.sleep(0.01)
        pos, orn = np.array(pose.translation()), np.array(pose.quaternion())
        return pos, orn

    def get_tcp_pose(self):
        return pos_orn_to_matrix(*self.get_tcp_pos_orn())

    def open_gripper(self, blocking=False):
        self.gripper.open(blocking)

    def close_gripper(self, blocking=False):
        self.gripper.close(blocking)

    def _frankx_async_impedance_motion(self, target_pos, target_orn):
        """
        Start new async impedance motion. Do not call this directly.

        Args:
            target_pos: (x,y,z)
            target_orn: quaternion (x,y,z,w) | euler_angles (α,β,γ)
        """
        target_pose = to_affine(target_pos, target_orn) * NE_T_EE
        if self._is_active(ImpedanceMotion):
            self.current_motion.set_target(target_pose)
        else:
            if self.current_motion is not None:
                self.abort_motion()
            self.current_motion = self._new_impedance_motion()
            self.current_motion.set_target(target_pose)
            self.motion_thread = self.robot.move_async(self.current_motion)

    def _new_impedance_motion(self):
        """
        Create new frankx impedance motion with the params specified in config file.

        Returns:
            Impedance motion object.
        """
        if self.impedance_params.use_nullspace:
            return ImpedanceMotion(
                self.impedance_params.translational_stiffness,
                self.impedance_params.rotational_stiffness,
                self.impedance_params.nullspace_stiffness,
                self.impedance_params.q_d_nullspace,
                self.impedance_params.damping_xi,
            )
        else:
            return ImpedanceMotion(
                self.impedance_params.translational_stiffness, self.impedance_params.rotational_stiffness
            )

    def _frankx_async_lin_motion(self, target_pos, target_orn):
        """
        Start new Waypaint motion without impedance. Do not call this directly.

        Args:
            target_pos: (x,y,z)
            target_orn: quaternion (x,y,z,w) | euler_angles (α,β,γ)
        """
        target_pose = to_affine(target_pos, target_orn) * NE_T_EE
        if self._is_active(WaypointMotion):
            self.current_motion.set_next_waypoint(Waypoint(target_pose))
        else:
            if self.current_motion is not None:
                self.abort_motion()
            self.current_motion = WaypointMotion(
                [
                    Waypoint(target_pose),
                ],
                return_when_finished=False,
            )
            self.motion_thread = self.robot.move_async(self.current_motion)

    def _inverse_kinematics(self, target_pos, target_orn):
        """
        Find inverse kinematics solution with the ik solver specified in config file.

        Args:
            target_pos: cartesian target position (x,y,z).
            target_orn: cartesian target orientation, quaternion (x,y,z,w) | euler_angles (α,β,γ).

        Returns:
            Target joint angles in rad.
        """
        current_q = self.get_state()["joint_positions"]
        new_q = self.ik_solver.inverse_kinematics(target_pos, target_orn, current_q)
        return new_q

    def _is_active(self, motion):
        """Returns True if there is a currently active motion with the same type as motion."""
        return (
            self.current_motion is not None
            and isinstance(self.current_motion, motion)
            and self.motion_thread.is_alive()
        )

    def visualize_external_forces(self, canvas_width=500):
        """
        Display the external forces (x,y,z) and torques (a,b,c) of the tcp frame.

        Args:
            canvas_width: Display width in pixel.

        """
        canvas = np.ones((300, canvas_width, 3))
        forces = self.get_state()["force_torque"]
        contact = np.array(self.libfranka_params.contact_force_threshold)
        collision = np.array(self.libfranka_params.collision_force_threshold)
        left = 10
        right = canvas_width - left
        width = right - left
        height = 30
        y = 10
        for i, (lcol, lcon, f, ucon, ucol) in enumerate(zip(-collision, -contact, forces, contact, collision)):
            cv2.rectangle(canvas, [left, y], [right, y + height], [0, 0, 0], thickness=2)
            force_bar_pos = int(left + width * (f - lcol) / (ucol - lcol))
            cv2.line(canvas, [force_bar_pos, y], [force_bar_pos, y + height], thickness=4, color=[0, 0, 1])
            ucon_bar_pos = int(left + width * (ucon - lcol) / (ucol - lcol))
            cv2.line(canvas, [ucon_bar_pos, y], [ucon_bar_pos, y + height], thickness=2, color=[1, 0, 0])
            lcon_bar_pos = int(left + width * (lcon - lcol) / (ucol - lcol))
            cv2.line(canvas, [lcon_bar_pos, y], [lcon_bar_pos, y + height], thickness=2, color=[1, 0, 0])

            y += height + 10
        cv2.imshow("external_forces", canvas)
        cv2.waitKey(1)


@hydra.main(config_path="../conf", config_name="panda_teleop.yaml")
def main(cfg):
    robot = hydra.utils.instantiate(cfg.robot)
    robot.move_to_neutral()
    robot.close_gripper()
    time.sleep(1)
    print(robot.get_tcp_pose())
    exit()
    # print(robot.get_state()["gripper_opening_width"])
    # time.sleep(2)
    # robot.open_gripper()
    # time.sleep(1)
    # exit()
    # pos, orn = robot.get_tcp_pos_orn()
    # pos[0] += 0.2
    # pos[2] -= 0.1
    # # pos[2] -= 0.05
    # print("move")
    # robot.move_cart_pos_abs_ptp(pos, orn)
    # time.sleep(5)
    # print("done!")


if __name__ == "__main__":
    main()
