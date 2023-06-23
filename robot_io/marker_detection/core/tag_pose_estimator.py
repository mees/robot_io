import cv2
import numpy as np


class TagPoseEstimator(object):
    """
    Calculates the 3D coordinates of the visible keypoints of a AprilTag based calibration board using a OpenCV
    PnP algorithm.
    """

    def __init__(self, board, verbose=False):
        self.board = board
        self.verbose = verbose

    def estimate_relative_cam_pose(self, camera_intrinsic, camera_dist, points2d_cam, point_ids):
        """Estimates the relative camera pose between two cameras from some given point correspondences."""
        # Check inputs
        assert len(camera_intrinsic.shape) == 2, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic.shape[0] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic.shape[1] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert (
            np.abs((camera_intrinsic[0, 0] - camera_intrinsic[1, 1]) / camera_intrinsic[1, 1]) < 0.1
        ), "camera_intrinsic needs to be symmetric focal length wise."

        camera_dist = np.squeeze(np.array(camera_dist))
        assert len(camera_dist.shape) == 1, "camera_dist shape mismatch. Should be of length 4 or 5."
        assert camera_dist.shape[0] in [4, 5, 8, 12, 14]

        assert len(points2d_cam.shape) == 2, "points2d_cam0 shape mismatch. Should be Nx2"
        assert points2d_cam.shape[0] >= 4, "points2d_cam0 shape mismatch. Should be Nx2, with N>4"
        assert points2d_cam.shape[1] == 2, "points2d_cam0 shape mismatch. Should be Nx2"

        point_ids = np.squeeze(np.array(point_ids))
        assert len(point_ids.shape) == 1, "point_ids shape mismatch. Should be N"
        assert point_ids.shape[0] == points2d_cam.shape[0], "point_ids shape mismatch. Should be N"

        # select points that were detected
        object_points_det, points2d_cam = self.board.get_matching_objectpoints(point_ids, points2d_cam)
        if len(object_points_det) < 4:
            return None

        # # calculate PNP (to get an estimate for the 3D point location)
        success, r_rel, t_rel = cv2.solvePnP(
            np.expand_dims(object_points_det, 1),
            np.expand_dims(points2d_cam, 1),
            camera_intrinsic,
            distCoeffs=camera_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # # This function is BUGGY in OpenCV 3.3
        # success, r_rel, t_rel, inliers = cv2.solvePnPRansac(np.expand_dims(object_points_det, 1), np.expand_dims(points2d_cam, 1),
        #                                                     cameraMatrix=camera_intrinsic, distCoeffs=camera_dist,
        #                                                     # flags=cv2.SOLVEPNP_EPNP)
        #                                                     # flags=cv2.SOLVEPNP_DLS)
        #                                                     flags=cv2.SOLVEPNP_ITERATIVE)

        R, _ = cv2.Rodrigues(r_rel)
        points3d_pred = np.matmul(object_points_det, np.transpose(R)) + np.transpose(t_rel)

        return points3d_pred, R, t_rel
