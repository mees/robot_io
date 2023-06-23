import os
from pathlib import Path
import time

import hydra
import numpy as np

from robot_io.calibration.calibration_utils import (
    calculate_error,
    calibrate_static_cam_least_squares,
    save_calibration,
    visualize_frame_in_static_cam,
)
from robot_io.utils.utils import FpsController, matrix_to_pos_orn


def record_new_poses(robot, marker_detector, cfg, calib_poses_dir):
    """
    Move the robot to diverse poses with the VR controller such that the marker which is held by or attached to the
    gripper is visible from the static camera.
    Press button 1 on the VR controller to use the current pose for calibration.
    Save approximately 30 to 50 poses for good calibration results.
    Hold button 1 to end recording.

    Args:
        robot: Robot interface
        marker_detector: Marker detection library.
        cfg: Hydra config.
        calib_poses_dir: Path where to save poses.
    """
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)
    fps = FpsController(cfg.freq)

    robot.move_to_neutral()
    time.sleep(5)
    robot.close_gripper(blocking=True)

    tcp_poses = []
    marker_poses = []

    recorder = hydra.utils.instantiate(cfg.recorder, save_dir=calib_poses_dir)
    while True:
        fps.step()
        action, record_info = input_device.get_action()
        if record_info["hold_event"]:
            return tcp_poses, marker_poses

        if action is None:
            continue

        target_pos, target_orn, _ = action["motion"]
        robot.move_async_cart_pos_abs_lin(target_pos, target_orn)

        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            tcp_pose = robot.get_tcp_pose()
            tcp_poses.append(tcp_pose)
            marker_poses.append(marker_pose)
            recorder.step(tcp_pose, marker_pose, record_info)


def detect_marker_from_trajectory(robot, tcp_poses, marker_detector, cfg):
    """
    Move to previously recorded poses and estimate marker poses.

    Args:
        robot: Robot interface
        tcp_poses: Previously saved tcp poses as list of 4x4 matrices.
        marker_detector: Marker detection library.
        cfg: Hydra config.

    Returns:
        valid_tcp_poses (list): TCP poses where a marker has been detected.
        marker_poses (list): The detected marker poses.
    """
    input("Robot will move to recorded poses. Press any key to continue.")
    robot.move_to_neutral()
    time.sleep(5)
    robot.close_gripper(blocking=True)
    marker_poses = []
    valid_tcp_poses = []

    for i in range(len(tcp_poses)):
        target_pos, target_orn = matrix_to_pos_orn(tcp_poses[i])
        robot.move_cart_pos_abs_ptp(target_pos, target_orn)
        time.sleep(1.0)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            valid_tcp_poses.append(tcp_poses[i])
            marker_poses.append(marker_pose)

    return valid_tcp_poses, marker_poses


def load_recording(path):
    """
    Load a recording of tcp and marker poses.

    Args:
        path: Path to recording of saved poses.

    Returns:
        tcp_poses (list): TCP poses as list of 4x4 matrices.
        marker_poses (list): Detected marker poses as list of 4x4 matrices.
    """
    tcp_poses = []
    marker_poses = []

    for filename in path.glob("*.npz"):
        pose = np.load(filename)
        tcp_poses.append(pose["tcp_pose"])
        marker_poses.append(pose["marker_pose"])

    return tcp_poses, marker_poses


@hydra.main(config_path="../conf")
def main(cfg):
    """
    Calibrate the static camera by attaching a marker to the end-effector and recording marker poses with VR control.
    If `record_new_poses=true`, record new poses for calibration.

    Args:
        cfg: Hydra config.
    """
    cam = hydra.utils.instantiate(cfg.cam)
    marker_detector = hydra.utils.instantiate(cfg.marker_detector, cam=cam)
    robot = hydra.utils.instantiate(cfg.robot)
    calib_poses_dir = Path(f"{robot.name}_{marker_detector.cam.name}_calib_poses")

    if cfg.record_new_poses:
        record_new_poses(robot, marker_detector, cfg, calib_poses_dir)

    tcp_poses, _ = load_recording(calib_poses_dir)
    # this will move the robot to previously recorded poses
    tcp_poses, marker_poses = detect_marker_from_trajectory(robot, tcp_poses, marker_detector, cfg)

    T_robot_cam = calibrate_static_cam_least_squares(tcp_poses, marker_poses)
    save_calibration(robot.name, cam.name, "cam", "robot", T_robot_cam)

    # if calibration was successful, you should see the robot base coordinate axes at the right position.
    visualize_frame_in_static_cam(cam, np.linalg.inv(T_robot_cam))


if __name__ == "__main__":
    main()
