import logging

import cv2
import numpy as np
import open3d as o3d

from robot_io.utils.utils import get_git_root

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, resolution, crop_coords, resize_resolution, name):
        self.resolution = resolution
        self.crop_coords = crop_coords
        self.resize_resolution = resize_resolution
        self.name = name

    def get_image(self):
        """get the the current rgb and depth image as float32 numpy arrays"""
        rgb, depth = self._get_image()
        rgb = self._crop_and_resize(rgb)
        depth = self._crop_and_resize(depth)
        return rgb, depth

    def _get_image(self):
        raise NotImplementedError

    def _crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            img = img[c[0] : c[1], c[2] : c[3]]
        if self.resize_resolution is not None:
            interp = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
            img = cv2.resize(img, tuple(self.resize_resolution), interpolation=interp)
        return img

    def revert_crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            res = (c[3] - c[2], c[1] - c[0])
        else:
            res = self.resolution
        if self.resize_resolution is not None:
            interp = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
            img = cv2.resize(img, res, interpolation=interp)
        if self.crop_coords is not None:
            if len(img.shape) == 2:
                # case depth image
                new_img = np.zeros(self.resolution[::-1], dtype=img.dtype)
            else:
                # case rgb image
                new_img = np.zeros((*self.resolution[::-1], 3), dtype=img.dtype)
            new_img[c[0] : c[1], c[2] : c[3]] = img
            img = new_img
        return img

    def get_intrinsics(self):
        raise NotImplementedError

    def get_projection_matrix(self):
        raise NotImplementedError

    def get_camera_matrix(self):
        raise NotImplementedError

    def get_dist_coeffs(self):
        raise NotImplementedError

    def compute_pointcloud(self, depth_img, rgb_img=None, far_val=10, homogeneous=False):
        if depth_img.shape != self.resolution[::-1]:
            rgb_img = self.revert_crop_and_resize(rgb_img) if rgb_img is not None else None
            depth_img = self.revert_crop_and_resize(depth_img)
        u_crd, v_crd = np.where(np.logical_and(depth_img > 0, depth_img < far_val))
        intr = self.get_intrinsics()
        cx = intr["cx"]
        cy = intr["cy"]
        fx = intr["fx"]
        fy = intr["fy"]

        Z = depth_img[u_crd, v_crd]
        X = (v_crd - cx) * Z / fx
        Y = (u_crd - cy) * Z / fy

        pointcloud = np.stack((X, Y, Z), axis=1)
        if homogeneous:
            pointcloud = np.concatenate((pointcloud, np.ones((X.shape[0], 1))), axis=1)
        if rgb_img is not None:
            rgb_img = rgb_img.astype(np.float)
            color = rgb_img[u_crd, v_crd] / 255.0
            pointcloud = np.concatenate([pointcloud, color], axis=1)
        return pointcloud

    @staticmethod
    def view_pointcloud(pointcloud):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        if pointcloud.shape[1] == 6:
            pc.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6])
        elif pointcloud.shape[1] == 7:
            pc.colors = o3d.utility.Vector3dVector(pointcloud[:, 4:7])
        o3d.visualization.draw_geometries([pc])

    def project(self, X):
        if X.shape[0] == 3:
            if len(X.shape) == 1:
                X = np.append(X, 1)
            else:
                X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        x = self.get_projection_matrix() @ X
        result = np.round(x[0:2] / x[2]).astype(int)
        width, height = self.get_intrinsics()["width"], self.get_intrinsics()["height"]
        if not (0 <= result[0] < width and 0 <= result[1] < height):
            log.warning("Projected point outside of image bounds")
        return result[0], result[1]

    def deproject(self, point, depth, homogeneous=False):
        """
        Arguments:
            point: (x, y)
            depth: scalar or array, if array index with point
            homogeneous: boolean, return homogenous coordinates
        """

        point_mat = np.zeros((self.resize_resolution[1], self.resize_resolution[0]))
        point_mat[point[1], point[0]] = 1
        transformed_coords = self.revert_crop_and_resize(point_mat)
        y_candidates, x_candidates = np.where(transformed_coords == 1)
        y_transformed = y_candidates[len(y_candidates) // 2]
        x_transformed = x_candidates[len(x_candidates) // 2]
        point = (x_transformed, y_transformed)

        if not np.isscalar(depth) and depth.shape != self.resolution[::-1]:
            depth = self.revert_crop_and_resize(depth)

        intr = self.get_intrinsics()
        cx = intr["cx"]
        cy = intr["cy"]
        fx = intr["fx"]
        fy = intr["fy"]

        v_crd, u_crd = point

        if np.isscalar(depth):
            Z = depth
        else:
            Z = depth[u_crd, v_crd]
        if Z == 0:
            return None
        X = (v_crd - cx) * Z / fx
        Y = (u_crd - cy) * Z / fy
        if homogeneous:
            return np.array([X, Y, Z, 1])
        else:
            return np.array([X, Y, Z])

    @staticmethod
    def draw_point(img, point, color=(255, 0, 0)):
        img[point[1], point[0]] = color

    def get_extrinsic_calibration(self, robot_name):
        calib_folder = get_git_root(__file__) / "robot_io/calibration/calibration_files"
        calib_files = list(calib_folder.glob(f"{robot_name}_{self.name}*npy"))
        if len(calib_files) == 0:
            log.error(f"Calibration for {robot_name} and {self.name} does not exist.")
            raise FileNotFoundError
        newest_calib = sorted(calib_files, reverse=True)[0]
        logging.info(f"Using calibration: {newest_calib}")
        calib_extrinsic = np.load(newest_calib)
        assert calib_extrinsic.shape == (4, 4)
        return calib_extrinsic
