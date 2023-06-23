# roslib.load_manifest('my_package')
import sys
import time

import numpy as np
import roslib
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R


class Kinect4:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/rgb/image_raw", Image, self.callback_rgb)
        self.depth_sub = rospy.Subscriber("/depth_to_rgb/image_raw", Image, self.callback_depth)
        self.camera_info_sub = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.callback_camera_info)

        self.rgb = None
        self.depth = None
        self.dist_coeffs = None
        self.camera_matrix = None
        self.projection_matrix = None
        self.intrinsics = None
        rospy.init_node("Kinect4", anonymous=True)

    def callback_rgb(self, data):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_depth(self, data):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)

    def callback_camera_info(self, data):
        self.dist_coeffs = data.D
        self.projection_matrix = np.reshape(data.P, (3, 4))
        self.camera_matrix = self.projection_matrix[:, :3]
        self.intrinsics = {
            "cx": data.P[2],
            "cy": data.P[6],
            "fx": data.P[0],
            "fy": data.P[5],
            "width": data.width,
            "height": data.height,
        }
        self.camera_info_sub.unregister()

    def get_intrinsics(self):
        while self.intrinsics is None:
            time.sleep(0.1)
        return self.intrinsics

    def get_image(self, undistorted=False):
        while self.rgb is None or self.depth is None or self.camera_matrix is None:
            time.sleep(0.1)
        if undistorted:
            rgb = cv2.undistort(self.rgb, self.camera_matrix, self.dist_coeffs)
            depth = cv2.undistort(self.depth, self.camera_matrix, self.dist_coeffs)
            return rgb, depth
        else:
            return self.rgb, self.depth

    def detect_marker(
        self, rgb, marker_id, marker_size, camera_matrix, dist_coeffs=np.zeros(12), marker_dict=aruco.DICT_4X4_250
    ):
        # rgb = rgb.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(marker_dict)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        # lists of ids and the corners belonging to each marker_id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if ids is not None and marker_id in ids:
            pos = np.where(ids == marker_id)[0][0]
            corners = corners[pos : pos + 1]
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            # (rvec-tvec).any() # get rid of that nasty numpy value array error

            # draw axis for the aruco markers
            aruco.drawAxis(rgb, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(rgb, corners)

            # code to show ids of the marker found
            strg = ""
            for i in range(0, ids.size):
                strg += str(ids[i][0]) + ", "

            cv2.putText(rgb, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print("marker detected with marker_id {}".format(strg))

        else:
            print("no marker detected")
            # code to show 'No Ids' when no markers are found
            cv2.putText(rgb, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("win2", rgb)
        cv2.waitKey(1)


def main(args):
    kinect = Kinect4()
    kinect.get_image()
    time.sleep(1)
    np.savez(
        "kinect4_params_720p.npz",
        dist_coeffs=kinect.dist_coeffs,
        projection_matrix=kinect.projection_matrix,
        camera_matrix=kinect.camera_matrix,
        intrinsics=kinect.intrinsics,
    )

    while 1:
        rgb, depth = kinect.get_image(undistorted=True)
        # depth -= 0.7
        # depth/=0.2
        # d = cv2.medianBlur(depth, 5)

        kinect.detect_marker(
            rgb, 1, 0.12, kinect.camera_matrix[:, :3], kinect.dist_coeffs, marker_dict=aruco.DICT_5X5_250
        )
        cv2.imshow("win", depth)
        cv2.waitKey(1)


#
if __name__ == "__main__":
    main(sys.argv)
