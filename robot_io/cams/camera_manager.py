from functools import partial
import signal
import sys
import time

import cv2
import hydra
import numpy as np

from robot_io.cams.threaded_camera import ThreadedCamera
from robot_io.utils.utils import upscale


def destroy_on_signal(self, sig, frame):
    # Stop streaming
    print(f"Received signal {signal.Signals(sig).name}. Exit cameras.")
    if isinstance(self.gripper_cam, ThreadedCamera):
        self.gripper_cam._camera_thread.flag_exit = True
        self.gripper_cam._camera_thread.join()
    if isinstance(self.static_cam, ThreadedCamera):
        self.static_cam._camera_thread.flag_exit = True
        self.static_cam._camera_thread.join()
    print("done")
    sys.exit()


class CameraManager:
    """
    Class for handling different cameras
    """

    def __init__(self, use_gripper_cam, use_static_cam, gripper_cam, static_cam, threaded_cameras, robot_name=None):
        self.gripper_cam = None
        self.static_cam = None
        if use_gripper_cam:
            if threaded_cameras:
                self.gripper_cam = ThreadedCamera(gripper_cam)
            else:
                self.gripper_cam = hydra.utils.instantiate(gripper_cam)
        if use_static_cam:
            if threaded_cameras:
                self.static_cam = ThreadedCamera(static_cam)
            else:
                self.static_cam = hydra.utils.instantiate(static_cam)
        self.obs = None
        self.robot_name = robot_name
        if robot_name is not None:
            self.save_calibration()
        signal.signal(signal.SIGINT, partial(destroy_on_signal, self))

    def get_images(self):
        obs = {}
        if self.gripper_cam is not None:
            rgb_gripper, depth_gripper = self.gripper_cam.get_image()
            obs["rgb_gripper"] = rgb_gripper
            obs["depth_gripper"] = depth_gripper
        if self.static_cam is not None:
            rgb, depth = self.static_cam.get_image()
            obs[f"rgb_static"] = rgb
            obs[f"depth_static"] = depth
        self.obs = obs
        return obs

    def save_calibration(self):
        camera_info = {}
        if self.gripper_cam is not None:
            camera_info["gripper_extrinsic_calibration"] = self.gripper_cam.get_extrinsic_calibration(self.robot_name)
            camera_info["gripper_intrinsics"] = self.gripper_cam.get_intrinsics()
        if self.static_cam is not None:
            camera_info["static_extrinsic_calibration"] = self.static_cam.get_extrinsic_calibration(self.robot_name)
            camera_info["static_intrinsics"] = self.static_cam.get_intrinsics()
        if len(camera_info):
            np.savez("camera_info.npz", **camera_info)

    def normalize_depth(self, img):
        img_mask = img == 0
        # we do not take the max because of outliers (wrong measurements)
        istats = (np.min(img[img > 0]), np.percentile(img, 95))

        imrange = (np.clip(img.astype("float32"), istats[0], istats[1]) - istats[0]) / (istats[1] - istats[0])
        imrange[img_mask] = 0

        imrange = 255.0 * imrange
        imsz = imrange.shape
        nchan = 1
        if len(imsz) == 3:
            nchan = imsz[2]
        imgcanvas = np.zeros((imsz[0], imsz[1], nchan), dtype="uint8")
        imgcanvas[0 : imsz[0], 0 : imsz[1]] = imrange.reshape((imsz[0], imsz[1], nchan))
        return imgcanvas

    def render(self):
        if "rgb_gripper" in self.obs:
            cv2.imshow("rgb_gripper", upscale(self.obs["rgb_gripper"][:, :, ::-1]))
        if "depth_gripper" in self.obs:
            depth_img_gripper = self.normalize_depth(self.obs["depth_gripper"])
            depth_img_gripper = cv2.applyColorMap(depth_img_gripper, cv2.COLORMAP_JET)
            cv2.imshow("depth_gripper", upscale(depth_img_gripper))
        if "rgb_static" in self.obs:
            cv2.imshow("rgb_static", upscale(self.obs["rgb_static"][:, :, ::-1]))
        cv2.waitKey(1)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    hydra.initialize("../conf/cams")
    cfg = hydra.compose("camera_manager.yaml")
    cam_manager = hydra.utils.instantiate(cfg)
    while True:
        cam_manager.get_images()
        cam_manager.render()
        time.sleep(0.05)
