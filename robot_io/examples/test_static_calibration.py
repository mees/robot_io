import time

import cv2
import hydra
import numpy as np


def get_point_in_world_frame(cam, robot, T_world_cam, clicked_point, depth):
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    point_world_frame = T_world_cam @ point_cam_frame
    return point_world_frame[:3]


@hydra.main(config_path="../conf", config_name="panda_calibrate_static_cam")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    robot = hydra.utils.instantiate(cfg.robot)
    T_world_cam = cam.get_extrinsic_calibration(robot.name)
    robot.move_to_neutral()
    pos, orn = robot.get_tcp_pos_orn()
    clicked_point = None

    def callback(event, x, y, flags, param):
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", callback)

    while 1:
        rgb, depth = cam.get_image()
        cv2.imshow("image", rgb[:, :, ::-1])
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
        if clicked_point is not None:
            print(clicked_point)
            point_world = get_point_in_world_frame(cam, robot, T_world_cam, clicked_point, depth)
            print(point_world)
            if point_world is not None:
                robot.move_cart_pos_abs_lin(point_world + np.array([0, 0, 0.02]), orn)
                robot.move_cart_pos_abs_lin(point_world, orn)
                robot.close_gripper()
                time.sleep(1)
                robot.move_to_neutral()
                robot.open_gripper()
            clicked_point = None


if __name__ == "__main__":
    main()
