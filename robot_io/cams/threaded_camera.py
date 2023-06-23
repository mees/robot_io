import logging
import multiprocessing
from multiprocessing import Process
import threading
import time

import hydra
import numpy as np

from robot_io.utils.utils import FpsController, timeit

log = logging.getLogger(__name__)


class ThreadedCamera:
    def __init__(self, camera_cfg):
        self._camera_thread = _CameraThread(camera_cfg)
        self._camera_thread.start()

    def get_image(self):
        while self._camera_thread.rgb is None or self._camera_thread.depth is None:
            time.sleep(0.01)
        rgb = self._camera_thread.rgb.copy()
        depth = self._camera_thread.depth.copy()

        return rgb, depth

    def get_crop_coords(self):
        return self._camera_thread.camera.crop_coords

    def get_resolution(self):
        return self._camera_thread.camera.resolution

    def get_resize_res(self):
        return self._camera_thread.camera.resize_resolution

    def get_intrinsics(self):
        return self._camera_thread.camera.get_intrinsics()

    def get_projection_matrix(self):
        return self._camera_thread.camera.get_projection_matrix()

    def get_camera_matrix(self):
        return self._camera_thread.camera.get_camera_matrix()

    def get_dist_coeffs(self):
        return self._camera_thread.camera.get_dist_coeffs()

    def compute_pointcloud(self, depth_img, rgb_img=None, far_val=10, homogeneous=False):
        return self._camera_thread.camera.compute_pointcloud(depth_img, rgb_img, far_val, homogeneous)

    def view_pointcloud(self, pointcloud):
        return self._camera_thread.camera.view_pointcloud(pointcloud)

    def revert_crop_and_resize(self, img):
        return self._camera_thread.camera.revert_crop_and_resize(img)

    def get_extrinsic_calibration(self, robot_name):
        return self._camera_thread.camera.get_extrinsic_calibration(robot_name)

    def deproject(self, point, depth, homogeneous=False):
        return self._camera_thread.camera.deproject(point, depth, homogeneous)

    def project(self, x):
        return self._camera_thread.camera.project(x)


class _CameraThread(threading.Thread):
    def __init__(self, camera_cfg):
        threading.Thread.__init__(self)
        self.camera = hydra.utils.instantiate(camera_cfg)
        self.daemon = True
        self.fps_controller = FpsController(camera_cfg.fps)
        self.flag_exit = False
        self.rgb = None
        self.depth = None

    def run(self):
        while not self.flag_exit:
            self.rgb, self.depth = self.camera.get_image()
            self.fps_controller.step()
        log.info("Exit camera thread.")


if __name__ == "__main__":
    import cv2
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("../conf/cams/gripper_cam/framos_highres.yaml")
    cam = ThreadedCamera(cfg)

    while True:
        rgb, depth = cam.get_image()
        # pc = cam.compute_pointcloud(depth, rgb)
        # cam.view_pointcloud(pc)
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)
