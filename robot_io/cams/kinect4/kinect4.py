from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import open3d as o3d

from robot_io.cams.camera import Camera


class Kinect4(Camera):
    def __init__(
        self,
        device=0,
        align_depth_to_color=True,
        resolution="1080p",
        undistort_image=True,
        resize_resolution=None,
        crop_coords=None,
        fps=30,
    ):

        config_path = "config/config_kinect4_{}.json".format(resolution)
        config = self.load_config_path(config_path)

        self.sensor = o3d.io.AzureKinectSensor(config)
        f = io.StringIO()
        with redirect_stdout(f):
            if not self.sensor.connect(device):
                raise RuntimeError("Failed to connect to sensor")
        device_info = f.getvalue()
        kinect_instance = self.get_kinect_instance(device_info)
        params_file_path = "config/kinect4{}_params_{}.npz".format(kinect_instance, resolution)
        resolution, data = self.load_config_data(params_file_path)

        super().__init__(
            resolution=resolution, crop_coords=crop_coords, resize_resolution=resize_resolution, name="azure_kinect"
        )
        self.dist_coeffs = data["dist_coeffs"]
        self.camera_matrix = data["camera_matrix"]
        self.projection_matrix = data["projection_matrix"]
        self.intrinsics = data["intrinsics"].item()
        self.intrinsics.update(
            {
                "crop_coords": self.crop_coords,
                "resize_resolution": self.resize_resolution,
                "dist_coeffs": self.dist_coeffs,
            }
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            R=np.eye(3),
            newCameraMatrix=self.camera_matrix,
            size=(self.intrinsics["width"], self.intrinsics["height"]),
            m1type=cv2.CV_16SC2,
        )
        self.device = device
        self.align_depth_to_color = align_depth_to_color
        self.config_path = config_path
        self.undistort_image = undistort_image
        self.fps = fps
        self.align_depth_to_color = align_depth_to_color

    def load_config_path(self, config_path):
        if config_path is not None:
            full_path = (Path(__file__).parent / config_path).as_posix()
            config = o3d.io.read_azure_kinect_sensor_config(full_path)
        else:
            config = o3d.io.AzureKinectSensorConfig()

        return config

    def load_config_data(self, params_file_path):
        data = np.load((Path(__file__).parent / params_file_path).as_posix(), allow_pickle=True)
        if "1080" in params_file_path:
            resolution = (1920, 1080)
        elif "720" in params_file_path:
            resolution = (1280, 720)
        else:
            raise ValueError

        return resolution, data

    def get_kinect_instance(self, device_info):
        serial_number = [int(s) for s in device_info.splitlines()[2].split() if s.isdigit()][0]
        if serial_number == 232793712:
            kinect_instance = "a"
        elif serial_number == 172402712:
            kinect_instance = "b"
        else:
            raise ValueError

        return kinect_instance

    def get_intrinsics(self):
        return self.intrinsics

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_camera_matrix(self):
        return self.camera_matrix

    def get_dist_coeffs(self):
        return self.dist_coeffs

    def _get_image(self):
        rgbd = None
        while rgbd is None:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        rgb = np.asarray(rgbd.color)
        depth = (np.asarray(rgbd.depth)).astype(np.float32) / 1000
        if self.undistort_image:
            rgb = cv2.remap(rgb, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return rgb, depth


def run_camera():
    cam = Kinect4(0)
    print(cam.get_intrinsics())
    while True:
        rgb, depth = cam.get_image()
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)


if __name__ == "__main__":
    run_camera()
