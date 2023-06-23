from pathlib import Path

import cv2
import hydra.utils
import numpy as np
from omegaconf import OmegaConf

from robot_io.cams.realsense.realsense import Realsense
from robot_io.marker_detection.core.board_detector import BoardDetector
from robot_io.marker_detection.core.tag_pose_estimator import TagPoseEstimator

# from cv2 import aruco


class ApriltagDetector:
    def __init__(self, cam, marker_description, min_tags):
        # set up detector and estimator
        self.cam = cam
        marker_description = (Path(__file__).parent / marker_description).as_posix()
        self.detector = BoardDetector(marker_description)
        self.min_tags = min_tags
        self.estimator = TagPoseEstimator(self.detector)
        self.K = cam.get_camera_matrix()
        self.dist_coeffs = np.zeros(12)

    def estimate_pose(self, rgb=None, visualize=True):
        if rgb is None:
            rgb, _ = self.cam.get_image()

        points2d, point_ids = self.detector.process_image_m(rgb)

        T_cam_marker = self._estimate_pose(points2d, point_ids)
        if T_cam_marker is None:
            print("No marker detected")
            return None
        if visualize:
            self.detector.draw_board(rgb, points2d, point_ids, show=False, linewidth=1)
            # aruco.drawAxis(rgb, self.K, self.dist_coeffs, T_cam_marker[:3, :3], T_cam_marker[:3, 3], 0.1)
            cv2.imshow("window", rgb[:, :, ::-1])
            cv2.waitKey(1)
        return T_cam_marker

    def _estimate_pose(self, p2d, pid):
        if p2d.shape[0] < self.min_tags * 4:
            return None
        ret = self.estimator.estimate_relative_cam_pose(self.K, self.dist_coeffs, p2d, pid)
        if ret is None:
            return None
        points3d_pred, rot, trans = ret
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = rot
        T_cam_marker[:3, 3:] = trans
        return T_cam_marker


if __name__ == "__main__":
    cam_cfg = OmegaConf.load("../conf/cams/gripper_cam/realsense.yaml")
    cam = hydra.utils.instantiate(cam_cfg)
    cfg = OmegaConf.load("../conf/marker_detector/apriltag_board.yaml")
    marker_detector = ApriltagDetector(cam, cfg.marker_description, cfg.min_tags)
    print("entering loop")
    while True:
        marker_detector.estimate_pose()
