import math
import time

import hydra
import numpy as np

from robot_io.calibration.calibration_utils import (
    calculate_error,
    calibrate_gripper_cam_least_squares,
    calibrate_gripper_cam_peak_martin,
    save_calibration,
    visualize_calibration_gripper_cam,
)
from robot_io.utils.utils import euler_to_quat, quat_to_euler


class GripperCamPoseSampler:
    """
    Randomly sample end-effector poses for gripper cam calibration.
    Poses are sampled with polar coordinates theta and r around initial_pos, which are then perturbed with a random
    positional and rotational offset

    Args:
        initial_pos: TCP position around which poses are sampled
        initial_orn: TCP orientation around which poses are sampled
        theta_limits: Angle for polar coordinate sampling wrt. X-axis in robot base frame
        r_limits: Radius for plar coordinate sampling
        h_limits: Sampling range for height offset
        trans_limits: Sampling range for lateral offset
        yaw_limits: Sampling range for yaw offset
        pitch_limit: Sampling range for pitch offset
        roll_limit: Sampling range for roll offset
    """

    def __init__(
        self,
        initial_pos,
        initial_orn,
        theta_limits,
        r_limits,
        h_limits,
        trans_limits,
        yaw_limits,
        pitch_limit,
        roll_limit,
    ):
        self.initial_pos = np.array(initial_pos)
        self.initial_orn = quat_to_euler(np.array(initial_orn))
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.yaw_limits = yaw_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit

    def sample_pose(self):
        """
        Sample a random pose
        Returns:
            target_pos: Position (x,y,z)
            target_pos: Orientation quaternion (x,y,z,w)
        """
        theta = np.random.uniform(*self.theta_limits)
        vec = np.array([np.cos(theta), np.sin(theta), 0])
        vec = vec * np.random.uniform(*self.r_limits)
        yaw = np.random.uniform(*self.yaw_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        height = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        trans_final = self.initial_pos + vec + trans + height
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)

        target_pos = np.array(trans_final)
        target_orn = np.array([math.pi + pitch, roll, theta + math.pi + yaw])
        target_orn = euler_to_quat(target_orn)
        return target_pos, target_orn


def record_gripper_cam_trajectory(robot, marker_detector, cfg):
    """
    Move robot to randomly generated poses and estimate marker poses.

    Args:
        robot: Robot interface.
        marker_detector: Marker detection library.
        cfg: Hydra config.

    Returns:
        tcp_poses (list): TCP poses as list of 4x4 matrices.
        marker_poses (list): Detected marker poses as list of 4x4 matrices.
    """
    robot.move_to_neutral()
    time.sleep(2)
    _, orn = robot.get_tcp_pos_orn()
    pose_sampler = hydra.utils.instantiate(cfg.gripper_cam_pose_sampler, initial_orn=orn)

    i = 0
    tcp_poses = []
    marker_poses = []

    while i < cfg.num_poses:
        pos, orn = pose_sampler.sample_pose()
        robot.move_cart_pos_abs_ptp(pos, orn)
        time.sleep(0.3)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            tcp_poses.append(robot.get_tcp_pose())
            marker_poses.append(marker_pose)
            i += 1

    return tcp_poses, marker_poses


@hydra.main(config_path="../conf", config_name="panda_calibrate_gripper_cam")
def main(cfg):
    """
    Calibrate the gripper camera.
    Put the marker on the table such that it is visible in the gripper camera from the randomly sampled poses.

    Args:
        cfg: Hydra config.
    """
    cam = hydra.utils.instantiate(cfg.cam)
    marker_detector = hydra.utils.instantiate(cfg.marker_detector, cam=cam)
    robot = hydra.utils.instantiate(cfg.robot)
    tcp_poses, marker_poses = record_gripper_cam_trajectory(robot, marker_detector, cfg)
    # np.savez("data.npz", tcp_poses=tcp_poses, marker_poses=marker_poses)
    # data = np.load("data.npz")
    # tcp_poses = list(data["tcp_poses"])
    # marker_poses = list(data["marker_poses"])
    T_tcp_cam = calibrate_gripper_cam_least_squares(tcp_poses, marker_poses)

    save_calibration(robot.name, cam.name, "cam", "tcp", T_tcp_cam)
    calculate_error(T_tcp_cam, tcp_poses, marker_poses)
    visualize_calibration_gripper_cam(cam, T_tcp_cam)


if __name__ == "__main__":
    main()
