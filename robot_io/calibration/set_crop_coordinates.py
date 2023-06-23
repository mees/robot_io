import cv2
import hydra
import numpy as np


def set_crop_coordinates(cam, resolution=None):
    """
    Get the coordinates of an image region of interest, by drawing a selection rectangle.
    Copy the values into a camera config to use them as default.

    Args:
        cam: Camera (Kinect4 or Realsense / Framos).
        resolution: Resolution defining the aspect ratio of the cropped ROI (only the aspect ratio is used here).
    """
    while True:
        for i in range(10):
            rgb, depth = cam.get_image()
            cv2.imshow("rgb", rgb[:, :, ::-1])
            cv2.imshow("depth", depth)
            cv2.waitKey(1)
        r = cv2.selectROI("rgb", rgb[:, :, ::-1])
        if r == (0, 0, 0, 0):
            continue
        x, y = int(r[0]), int(r[1])
        width, height = int(r[2]), int(r[3])
        center = (x + width // 2, y + height // 2)
        ratio = resolution[0] / resolution[1] if resolution is not None else cam.resolution[0] / cam.resolution[1]
        if width / height > ratio:
            height = width / ratio
        else:
            width = height * ratio
        height = int(np.round(height))
        width = int(np.round(width))

        mask = np.zeros_like(depth)
        mask[center[1] - height // 2 : center[1] + height // 2, center[0] - width // 2 : center[0] + width // 2] = 1
        rgb[np.where(mask == 0)] = 0
        depth[np.where(mask == 0)] = 0

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("depth", depth)
        print("Press ENTER to finish selection, press c button to redo.")
        k = cv2.waitKey(0) % 256
        if k == 13:
            print(
                f"Image coordinates: ({center[1] - height // 2}, {center[1] + height // 2}, {center[0] - width // 2}, {center[0] + width // 2})"
            )
            return
        else:
            continue


@hydra.main(config_path="../conf", config_name="set_crop_coordinates")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    set_crop_coordinates(cam, resolution=cfg.resolution)


if __name__ == "__main__":
    main()
