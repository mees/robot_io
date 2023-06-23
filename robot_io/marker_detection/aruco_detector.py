import cv2
import cv2.aruco as aruco
import hydra.utils
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

FONT = cv2.FONT_HERSHEY_SIMPLEX


class ArucoDetector:
    def __init__(self, cam, marker_size, marker_dict, marker_id, visualize=True):
        self.cam = cam
        self.marker_size = marker_size
        self.visualize = visualize
        self.camera_matrix = self.cam.get_camera_matrix()
        self.marker_dict = eval(marker_dict) if isinstance(marker_dict, str) else marker_dict
        self.marker_id = marker_id
        self.dist_coeffs = self.cam.get_dist_coeffs()

    def detect_markers(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(self.marker_dict)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        # lists of ids and the corners belonging to each marker_id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None:
            return {}
        return {_id[0]: [_corners] for _id, _corners in zip(ids, corners)}

    def detect_all_markers(self):
        rgb, depth = self.cam.get_image()
        detected_markers = self.detect_markers(rgb)
        if len(detected_markers) == 0:
            print("no marker detected")
            # code to show 'No Ids' when no markers are found

            cv2.putText(rgb, "No Ids", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("win2", rgb[:, :, ::-1])
            cv2.waitKey(1)

            return False
        for _id, corners in detected_markers.items():
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            # (rvec-tvec).any() # get rid of that nasty numpy value array error

            # draw axis for the aruco markers
            aruco.drawAxis(rgb, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(rgb, corners)

        cv2.putText(rgb, f"Ids: {list(detected_markers.keys())}", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"marker detected with marker_ids {detected_markers.keys()}")
        #
        cv2.imshow("win2", rgb[:, :, ::-1])
        cv2.waitKey(1)

    def estimate_pose(self, rgb=None, marker_id=None):
        if marker_id is None:
            marker_id = self.marker_id
        if rgb is None:
            rgb, _ = self.cam.get_image()
        detected_markers = self.detect_markers(rgb)

        if len(detected_markers) == 0 or marker_id not in detected_markers:
            print("no marker detected")
            # code to show 'No Ids' when no markers are found
            if self.visualize:
                cv2.putText(rgb, "No Ids", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("win2", rgb[:, :, ::-1])
                cv2.waitKey(1)

            return None

        corners = detected_markers[marker_id]
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
        # (rvec-tvec).any() # get rid of that nasty numpy value array error

        if self.visualize:
            # draw axis for the aruco markers
            aruco.drawAxis(rgb, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(rgb, corners)

            cv2.putText(rgb, f"Id: {marker_id}", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"marker detected with marker_id {marker_id}")
            #
            cv2.imshow("win2", rgb[:, :, ::-1])
            cv2.waitKey(1)

        r = R.from_rotvec(rvec[0])
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, 3] = tvec
        T_cam_marker[:3, :3] = r.as_matrix()
        return T_cam_marker


if __name__ == "__main__":
    # from robot_io.cams.kinect4.kinect4 import Kinect4
    # cam = Kinect4()
    from robot_io.cams.realsense.realsense import Realsense

    cam = Realsense()
    cfg = OmegaConf.load("../conf/marker_detector/aruco.yaml")
    marker_detector = hydra.utils.instantiate(cfg, cam=cam)

    while True:
        print(marker_detector.estimate_pose())
