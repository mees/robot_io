from datetime import datetime
import itertools

import cv2
import numpy as np
from numpy import dot, eye, outer, zeros
from numpy.linalg import inv
from scipy.optimize import least_squares
from scipy.spatial.transform.rotation import Rotation as R

from robot_io.utils.utils import matrix_to_pos_orn, pos_orn_to_matrix


def pprint(arr):
    return np.array2string(arr.round(5), separator=", ")


def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * theta / (2 * np.sin(theta))


def invsqrt(mat):
    u, s, v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0 / np.sqrt(s))).dot(v)


def calibrate(A, B):
    # transform pairs A_i, B_i
    N = len(A)
    M = np.zeros((3, 3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3 * N, 3))
    d = zeros((3 * N, 1))
    for i in range(N):
        Ra, ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb, tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3 * i : 3 * i + 3, :] = eye(3) - Ra
        d[3 * i : 3 * i + 3, 0] = ta - dot(Rx, tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx.flatten()
    return X


def calibrate_gripper_cam_peak_martin(T_robot_tcp_list, T_cam_marker_list):
    ECs = []
    for T_robot_tcp, t_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        ECs.append((np.linalg.inv(T_robot_tcp), t_cam_marker))

    As = []  # relative EEs
    Bs = []  # relative cams
    for pair in itertools.combinations(ECs, 2):
        (e_1, c_1), (e_2, c_2) = pair
        A = e_2 @ np.linalg.inv(e_1)
        B = c_2 @ np.linalg.inv(c_1)
        As.append(A)
        Bs.append(B)

        # symmetrize
        A = e_1 @ np.linalg.inv(e_2)
        B = c_1 @ np.linalg.inv(c_2)
        As.append(A)
        Bs.append(B)

    X = calibrate(As, Bs)
    return X


def compute_residuals_gripper_cam(x, T_robot_tcp, T_cam_marker):
    m_R = np.array([*x[6:], 1])
    T_cam_tcp = pos_orn_to_matrix(x[3:6], x[:3])

    residuals = []
    for i in range(len(T_cam_marker)):
        m_C_observed = T_cam_marker[i][:3, 3]
        m_C = T_cam_tcp @ np.linalg.inv(T_robot_tcp[i]) @ m_R
        residuals += list(m_C_observed - m_C[:3])
    return residuals


def calibrate_gripper_cam_least_squares(T_robot_tcp, T_cam_marker):
    initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(
        fun=compute_residuals_gripper_cam, x0=initial_guess, method="lm", args=(T_robot_tcp, T_cam_marker)
    )
    trans = result.x[3:6]
    rot = result.x[0:3]
    T_cam_tcp = pos_orn_to_matrix(trans, rot)
    T_tcp_cam = np.linalg.inv(T_cam_tcp)
    return T_tcp_cam


def visualize_calibration_gripper_cam(cam, T_tcp_cam):
    T_cam_tcp = np.linalg.inv(T_tcp_cam)

    left_finger = np.array([0, 0.04, 0, 1])
    right_finger = np.array([0, -0.04, 0, 1])
    tcp = np.array([0, 0, 0, 1])
    x = np.array([0.03, 0, 0, 1])
    y = np.array([0, 0.03, 0, 1])
    z = np.array([0, 0, 0.03, 1])

    left_finger_cam = T_cam_tcp @ left_finger
    right_finger_cam = T_cam_tcp @ right_finger
    tcp_cam = T_cam_tcp @ tcp
    x_cam = T_cam_tcp @ x
    y_cam = T_cam_tcp @ y
    z_cam = T_cam_tcp @ z

    rgb, _ = cam.get_image()
    cv2.circle(rgb, cam.project(left_finger_cam), radius=4, color=(255, 0, 0), thickness=3)
    cv2.circle(rgb, cam.project(right_finger_cam), radius=4, color=(0, 255, 0), thickness=3)

    cv2.line(rgb, cam.project(tcp_cam), cam.project(x_cam), color=(255, 0, 0), thickness=3)
    cv2.line(rgb, cam.project(tcp_cam), cam.project(y_cam), color=(0, 255, 0), thickness=3)
    cv2.line(rgb, cam.project(tcp_cam), cam.project(z_cam), color=(0, 0, 255), thickness=3)

    cv2.imshow("calibration", rgb[:, :, ::-1])
    cv2.waitKey(0)


def compute_residuals_static_cam(x, T_robot_tcp, T_cam_marker):
    m_tcp = np.array([*x[6:], 1])
    T_cam_robot = pos_orn_to_matrix(x[3:6], x[:3])

    residuals = []
    for i in range(len(T_cam_marker)):
        m_C_observed = T_cam_marker[i][:3, 3]
        m_C = T_cam_robot @ T_robot_tcp[i] @ m_tcp
        residuals += list(m_C_observed - m_C[:3])
    return residuals


def calibrate_static_cam_least_squares(T_robot_tcp, T_cam_marker):
    initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(
        fun=compute_residuals_static_cam, x0=initial_guess, method="lm", args=(T_robot_tcp, T_cam_marker)
    )
    trans = result.x[3:6]
    rot = result.x[0:3]
    T_cam_robot = pos_orn_to_matrix(trans, rot)
    T_robot_cam = np.linalg.inv(T_cam_robot)
    return T_robot_cam


def visualize_frame_in_static_cam(cam, T_cam_object):

    object = np.array([0, 0, 0, 1])
    x = np.array([0.1, 0, 0, 1])
    y = np.array([0, 0.1, 0, 1])
    z = np.array([0, 0, 0.1, 1])

    object_cam = T_cam_object @ object
    x_cam = T_cam_object @ x
    y_cam = T_cam_object @ y
    z_cam = T_cam_object @ z

    rgb, _ = cam.get_image()

    cv2.line(rgb, cam.project(object_cam), cam.project(x_cam), color=(255, 0, 0), thickness=3)
    cv2.line(rgb, cam.project(object_cam), cam.project(y_cam), color=(0, 255, 0), thickness=3)
    cv2.line(rgb, cam.project(object_cam), cam.project(z_cam), color=(0, 0, 255), thickness=3)

    cv2.imshow("object in cam", rgb[:, :, ::-1])
    cv2.waitKey(0)


def save_calibration(robot_name, cam_name, frame_from, frame_to, data):
    now = datetime.now()
    file_name = f"{robot_name}_{cam_name}_T_{frame_to}_{frame_from}_{now.strftime('%Y_%m_%d__%H_%M')}.npy"
    np.save(file_name, data)
    print(f"saved calibration to {file_name}")


def calculate_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list):
    result = []
    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        T_robot_marker = T_robot_tcp @ T_tcp_cam @ T_cam_marker
        pos, orn = matrix_to_pos_orn(T_robot_marker)
        result.append(pos)
    print(np.std(result, axis=0))
    return result
