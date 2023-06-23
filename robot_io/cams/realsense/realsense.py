## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

import time

import hydra
import numpy as np
from omegaconf import OmegaConf

# Import the library
import pyrealsense2 as rs

from robot_io.cams.camera import Camera


class Realsense(Camera):
    """
    Interface class to get data from realsense camera (e.g. framos d435e).
    """

    def __init__(
        self,
        fps=30,
        img_type="rgb",
        resolution=(640, 480),
        resize_resolution=None,
        crop_coords=None,
        params=None,
        name=None,
    ):
        assert img_type in ["rgb", "rgb_depth"]
        self.img_type = img_type
        super().__init__(resolution=resolution, crop_coords=crop_coords, resize_resolution=resize_resolution, name=name)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)

        # Start streaming
        for i in range(5):
            try:
                self.profile = self.pipeline.start(config)
                break
            except RuntimeError as e:
                print(e)
                print("Retrying")
                time.sleep(2)
        else:
            raise RuntimeError
        self.color_sensor = self.profile.get_device().first_color_sensor()
        self.depth_scale = None
        if img_type == "rgb_depth":
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            self.depth_scale = self.depth_sensor.get_depth_scale()

        self.set_parameters(params)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def __del__(self):
        # Stop streaming
        print("exit realsense")
        self.pipeline.stop()

    def _get_image(self):
        """get the the current image as a numpy array"""
        # Wait for a coherent pair of frames: depth and color
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                break
            except RuntimeError as e:
                print(e)

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.img_type == "rgb":
            return color_image, None
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float32)
        depth_image *= self.depth_scale
        return color_image, depth_image

    def set_parameters(self, params):
        if params is not None:
            for key, value in params.items():
                self.color_sensor.set_option(getattr(rs.option, key), value)

    def get_intrinsics(self):
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        intr_rgb = dict(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            crop_coords=self.crop_coords,
            resize_resolution=self.resize_resolution,
            dist_coeffs=self.get_dist_coeffs(),
        )
        return intr_rgb

    def get_projection_matrix(self):
        intr = self.get_intrinsics()
        cam_mat = np.array([[intr["fx"], 0, intr["cx"], 0], [0, intr["fy"], intr["cy"], 0], [0, 0, 1, 0]])
        return cam_mat

    def get_camera_matrix(self):
        intr = self.get_intrinsics()
        cam_mat = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
        return cam_mat

    def get_dist_coeffs(self):
        return np.zeros(12)


def test_cam():
    # Import OpenCV for easy image rendering
    import cv2

    cam_cfg = OmegaConf.load("../../conf/cams/gripper_cam/framos_highres.yaml")
    cam = hydra.utils.instantiate(cam_cfg)

    rgb, depth = cam.get_image()
    print(cam.get_intrinsics())
    while 1:
        rgb, depth = cam.get_image()

        # pc = cam.compute_pointcloud(depth, rgb)
        # cam.view_pointcloud(pc)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        # depth *= (255 / 4)
        # depth = np.clip(depth, 0, 255)
        # depth = depth.astype(np.uint8)
        # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)


if __name__ == "__main__":
    test_cam()
