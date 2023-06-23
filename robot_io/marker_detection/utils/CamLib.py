"""" Small library providing commonly used operations with cameras. """
import cv2
import numpy as np


def triangulate_opencv(K1, dist1, M1, K2, dist2, M2, points1, points2, invert_M=False):
    """Triangulates a 3D point from 2D observations and calibration.
    K1, K2: Camera intrinsics
    M1, M2: Camera extrinsics, mapping from world -> cam coord if invert_M is False
    points1, points2: Nx2 np.array of observed points
    invert_M: flag, set true when given M1, M2 are the mapping from cam -> root
    """
    if invert_M:
        M1 = np.linalg.inv(M1)
        M2 = np.linalg.inv(M2)

    # assemble projection matrices
    P1 = np.matmul(K1, M1[:3, :])
    P2 = np.matmul(K2, M2[:3, :])

    # undistort points
    point_cam1_coord = undistort_points(points1, K1, dist1)
    point_cam2_coord = undistort_points(points2, K2, dist2)

    # triangulate to 3D
    points3d_h = cv2.triangulatePoints(P1, P2, np.transpose(point_cam1_coord), np.transpose(point_cam2_coord))
    points3d_h = np.transpose(points3d_h)
    return _from_hom(points3d_h)


def project(xyz_coords, K, dist=None):
    """Projects a (x, y, z) tuple of world coords into the camera image frame."""
    xyz_coords = np.reshape(xyz_coords, [-1, 3])
    uv_coords = np.matmul(xyz_coords, np.transpose(K, [1, 0]))
    uv_coords = _from_hom(uv_coords)
    if dist is not None:
        uv_coords = distort_points(uv_coords, K, dist)
    return uv_coords


def backproject(uv_coords, z_coords, K, dist=None):
    """Backprojects a (u, v) distorted point observation within the image frame into the corresponding world frame."""
    uv_coords = np.reshape(uv_coords, [-1, 2])
    z_coords = np.reshape(z_coords, [-1, 1])
    assert uv_coords.shape[0] == z_coords.shape[0], "Number of points differs."

    if dist is not None:
        uv_coords = undistort_points(uv_coords, K, dist)

    uv_coords_h = _to_hom(uv_coords)
    xyz_coords = z_coords * np.matmul(uv_coords_h, np.transpose(np.linalg.inv(K), [1, 0]))
    return xyz_coords


def trafo_coords(xyz_coords, T):
    """Applies a given a transformation T to a set of euclidean coordinates."""
    xyz_coords_h = _to_hom(xyz_coords)
    xyz_trafo_coords_h = np.matmul(xyz_coords_h, np.transpose(T, [1, 0]))
    return _from_hom(xyz_trafo_coords_h)


def _to_hom(coords):
    """Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]."""
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
    return coords_h


def _from_hom(coords_h):
    """Turns the homogeneous coordinates [N, D+1] into [N, D]."""
    coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
    return coords


def distort_points(points, K, dist):
    """Given points this function returns where the observed points would lie with lens distortion."""
    assert len(points.shape) == 2, "Shape mismatch."
    dist = np.reshape(dist, [5])
    points = points.copy()

    # To relative coordinates
    points[:, 0] = (points[:, 0] - K[0, 2]) / K[0, 0]
    points[:, 1] = (points[:, 1] - K[1, 2]) / K[1, 1]

    # squared radial distance
    r2 = points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]

    # get distortion params
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]

    # Radial distorsion
    dist_x = points[:, 0] * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    dist_y = points[:, 1] * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    x, y = points[:, 0], points[:, 1]
    dist_x += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dist_y += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Back to absolute coordinates.
    dist_x = dist_x * K[0, 0] + K[0, 2]
    dist_y = dist_y * K[1, 1] + K[1, 2]
    points = np.stack([dist_x, dist_y], 1)
    points = np.reshape(points, [-1, 2])
    return points


def undistort_points(points, K, dist):
    """Given observed points this function returns where the point would lie when there would be no lens distortion."""
    points = np.reshape(points, [-1, 2]).astype(np.float32)

    # Runs an iterative algorithm to invert what distort_points(..) does
    points_dist = cv2.undistortPoints(np.expand_dims(points, 0), K, np.squeeze(dist), P=K)
    return np.squeeze(points_dist, 0)


def _rectify_image_pair(
    image1,
    K1,
    dist1,
    M1,
    image2,
    K2,
    dist2,
    M2,
    image1_shift=None,
    image2_shift=None,
    swap_cams=False,
    cams_are_lr=True,
):
    """Given two images and their calibration this applies Fusiello's method to rectify the images."""
    # default values for the image offsets
    if image1_shift is None:
        image1_shift = np.zeros((2,))
    if image2_shift is None:
        image2_shift = np.zeros((2,))

    # extract R, t
    R1, t1 = M1[:3, :3], M1[:3, 3]
    R2, t2 = M2[:3, :3], M2[:3, 3]

    # calculate camera matrices
    P1 = np.matmul(K1, M1[:3, :])
    P2 = np.matmul(K2, M2[:3, :])

    # calculate optical centers
    c1 = -np.matmul(R1.transpose(), np.matmul(np.linalg.inv(K1), P1[:, 3]))
    c2 = -np.matmul(R2.transpose(), np.matmul(np.linalg.inv(K2), P2[:, 3]))

    if cams_are_lr:
        # new x axis (baseline between c1 and c2)
        v1 = c2 - c1

        # f = np.sign(np.dot(np.cross(R1[1, :], R1[2, :]), c2))  # is 1.0 if cam1 is left and cam2 is right, -1.0 otherwise
        # print('mysterious factor', f)

        # new y axes (orthogonal to old z and new x)
        v2 = np.cross(R1[2, :], v1)
        if swap_cams:
            v1 = c1 - c2
            v2 = np.cross(R2[2, :], v1)
        # new z axes (orthogonal to v1 and v2)
        v3 = np.cross(v1, v2)

    else:
        # new y axis (baseline between c1 and c2)
        v2 = c2 - c1
        # new x axes (orthogonal to old z and new y)
        v1 = np.cross(v2, R1[2, :])
        if swap_cams:
            v2 = c1 - c2
            v1 = np.cross(v2, R2[2, :])
        # new z axes (orthogonal to v1 and v2)
        v3 = np.cross(v1, v2)

    # catch corner case where the cameras share an optical center
    if np.sqrt(np.sum(np.square(v1))) < 1e-6:
        R = np.eye(3)
    else:

        def _normed(x):
            return x / np.sqrt(np.sum(np.square(x)))

        # new extrinsics
        R = np.stack([_normed(v1), _normed(v2), _normed(v3)], 0)

    # new intrinsics
    K_new1 = K2.copy()
    K_new2 = K2.copy()
    K_new1[0, 1] = 0.0  # remove skew
    K_new2[0, 1] = 0.0
    K_new1[:2, 2] += image1_shift
    K_new2[:2, 2] += image2_shift

    # new projection matrices
    M1_new = np.concatenate([R, -np.expand_dims(np.matmul(R, c1), 1)], 1)
    P1_new = np.matmul(K_new1, M1_new)

    M2_new = np.concatenate([R, -np.expand_dims(np.matmul(R, c2), 1)], 1)
    P2_new = np.matmul(K_new2, M2_new)

    # rectifying image transformation
    T1 = np.matmul(P1_new[:3, :3], np.linalg.inv(P1[:3, :3]))
    T2 = np.matmul(P2_new[:3, :3], np.linalg.inv(P2[:3, :3]))

    return T1, T2


def can_see(point3d, M, K, dist, image_shape):
    """Check if points are visible in a camera or not."""
    point3d = np.reshape(point3d, [-1, 3])
    p2d = project(trafo_coords(point3d, M), K, dist)
    return np.logical_and(0.0 <= p2d[:, 0] < image_shape[1] - 1, 0.0 <= p2d[:, 1] < image_shape[0] - 1)


def rectify_image_pair_old(image1, K1, M1, image2, K2, M2, dist1=None, dist2=None):
    """Given two images and their calibration this applies Fusiello's method to rectify the images.
    Inputs:
        images as np.uint8
        K1, K2 as 3x3 matrices
        M1, M2 as 4x4 matrices describing the trafo from world to cam.
        dist1, dist2 as 1x5 matrices containing OpenCV distortion parameters.
                If none are given assumes image1 and image2 are already undistorted.
    Returns:
        cams_are_lr: If true the two images are horizontally aligned, else they are vertically aligned.
    """
    # figure out how cameras are roughly oriented
    M_two2one = np.matmul(M1, np.linalg.inv(M2))  # cam1 ... trafo(world to cam1) * trafo(cam2 to world) ... cam2
    t_two2one = M_two2one[
        :3, 3
    ]  # if you take 0 (in cam2 coords) and trafo it this tells you where 2 is from 1's perspective
    M_one2two = np.linalg.inv(M_two2one)  # trafo from cam1 to cam2
    t_one2two = M_one2two[:3, 3]  # where cam1 is from 2's view

    # detect if it is even possible to rectify (if the cameras mostly moved in z direction)
    see1 = can_see(t_two2one, np.eye(4), K1, dist1, image1.shape[:2])
    see2 = can_see(t_one2two, np.eye(4), K2, dist2, image2.shape[:2])

    if see1 or see2:
        raise Exception("Cameras can see each others centrums. Impossible to rectify.")

    cams_are_lr = False
    swap_cams = False
    if np.abs(t_two2one[0]) > np.abs(t_two2one[1]):
        # cameras are left/right
        cams_are_lr = True
        if t_two2one[0] < 0:
            # cam 2 is to the 'right' of cam 1
            swap_cams = True  # this is sematically wrong
    else:
        # cameras are top/down
        cams_are_lr = False
        if t_two2one[1] < 0:
            # cam 2 is 'above' of cam 1
            swap_cams = True
    print("vec cam2 in 1", t_two2one, cams_are_lr, swap_cams)

    # rectify without centering
    T1, T2 = _rectify_image_pair(
        image1, K1, dist1, M1, image2, K2, dist2, M2, swap_cams=swap_cams, cams_are_lr=cams_are_lr
    )

    # calculate new image centers: we center on the principal point
    p = np.array([K1[0, 2], K1[1, 2], 1.0])
    p_t = np.matmul(T1, np.expand_dims(p, 1)).squeeze()
    offset1 = p[:2] - p_t[:2] / p_t[-1]  # how much pp1 moved according to T1
    # c = np.array([image1.shape[1] / 2.0, image1.shape[0] / 2.0])
    # offset1 = c - p_t

    # calculate new image centers
    p = np.array([K2[0, 2], K2[1, 2], 1.0])
    p_t = np.matmul(T2, np.expand_dims(p, 1)).squeeze()
    offset2 = p[:2] - p_t[:2] / p_t[-1]  # how much pp2 moved according to T2
    # c = np.array([image2.shape[1] / 2.0, image2.shape[0] / 2.0])
    # offset2 = c - p_t

    if cams_are_lr:
        # make vertical offset equal
        offset1[1] = offset2[1]
    else:
        # make horizontal offset equal
        offset1[0] = offset2[0]

    # rectify with centering
    T1, T2 = _rectify_image_pair(
        image1, K1, dist1, M1, image2, K2, dist2, M2, offset1, offset2, swap_cams=swap_cams, cams_are_lr=cams_are_lr
    )

    # Undistort images if necessary
    if (dist1 is not None) and (dist2 is not None):
        image1 = cv2.undistort(image1, K1, dist1, newCameraMatrix=K1)
        image2 = cv2.undistort(image2, K2, dist2, newCameraMatrix=K2)

    # warp images with the found homography
    img_shape = image1.shape[:2]
    img_shape = img_shape[::-1]
    img_rect1 = cv2.warpPerspective(image1, T1, img_shape)
    img_rect2 = cv2.warpPerspective(image2, T2, img_shape)

    return cams_are_lr, swap_cams, img_rect1, img_rect2


def _rot90_around_pp(image, K):
    pp = (K[0, 2], K[1, 2])
    # get a rotation matrix that rotates around the pp
    M_rot = cv2.getRotationMatrix2D(pp, 90.0, 1.0)

    # translate by difference of principal points
    d_x = pp[0] - pp[1]
    d_y = pp[1] - pp[0]
    M_trans = np.array([[1.0, 0.0, -d_x], [0.0, 1.0, -d_y], [0.0, 0.0, 1.0]])

    # concat the two trafo
    M_rot = np.concatenate([M_rot, np.array([[0.0, 0.0, 1.0]])], 0)
    M = np.matmul(M_trans, M_rot)

    # perform warp with combined trafo
    image = cv2.warpAffine(image, M[:2, :], image.shape[:2])
    return image, pp


def rotate_images_90deg(image1, K1, M1, image2, K2, M2, dist1=None, dist2=None):
    """
    Rotates given images by 90 deg counter clock wise and adapts extrinsics accordingly.
    """
    show = False

    if show:
        # calculate some 3D points
        p1 = np.array([[721, 425.0], [983, 472.0], [390, 245.0], [127, 683.0]])
        p2 = np.array([[739, 428.0], [1024, 489.0], [438.0, 244], [134, 663]])
        p3ds = triangulate_opencv(K1, dist1, M1, K2, dist2, M2, p1, p2)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        p1 = project(trafo_coords(p3ds, M1), K1, dist1)
        p2 = project(trafo_coords(p3ds, M2), K2, dist2)
        ax1.imshow(image1)
        ax2.imshow(image2)
        for i in range(p1.shape[0]):
            ax1.scatter(p1[i, 0], p1[i, 1])
            ax2.scatter(p2[i, 0], p2[i, 1])

    # rotate
    R = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    image1, pp1 = _rot90_around_pp(image1, K1)
    M1 = np.matmul(R, M1)
    image2, pp2 = _rot90_around_pp(image2, K2)
    M2 = np.matmul(R, M2)

    # change the cams focal lengths and pricipal points
    tmp = K1.copy()
    tmp[0, 0] = K1[1, 1]
    tmp[0, 2] = K1[1, 2]
    tmp[1, 1] = K1[0, 0]
    tmp[1, 2] = K1[0, 2]
    K1 = tmp.copy()

    tmp = K2.copy()
    tmp[0, 0] = K2[1, 1]
    tmp[0, 2] = K2[1, 2]
    tmp[1, 1] = K2[0, 0]
    tmp[1, 2] = K2[0, 2]
    K2 = tmp.copy()

    if show:
        p1 = project(trafo_coords(p3ds, M1), K1)
        p2 = project(trafo_coords(p3ds, M2), K2)
        ax3.imshow(image1)
        ax4.imshow(image2)
        for i in range(p1.shape[0]):
            ax3.scatter(p1[i, 0], p1[i, 1])
            ax4.scatter(p2[i, 0], p2[i, 1])
        plt.show()
    return image1, K1, M1, image2, K2, M2


def _calc_rect_homog_lr(K1, M1, K2, M2, image_shift=None):
    """Given two undistorted images in a left (cam1) righ (cam2) setting and
    their calibration this applies Fusiello's method to find homographies for rectification."""
    # default values for the image offsets
    if image_shift is None:
        image_shift = np.zeros((2,))

    # extract R, t
    R1, t1 = M1[:3, :3], M1[:3, 3]
    R2, t2 = M2[:3, :3], M2[:3, 3]

    # calculate camera matrices
    P1 = np.matmul(K1, M1[:3, :])
    P2 = np.matmul(K2, M2[:3, :])

    # calculate optical centers in cam relative coords
    c1 = -np.matmul(R1.transpose(), np.matmul(np.linalg.inv(K1), P1[:, 3]))
    c2 = -np.matmul(R2.transpose(), np.matmul(np.linalg.inv(K2), P2[:, 3]))

    # new x axis (baseline between c1 and c2)
    v1 = c2 - c1
    f = np.sign(np.dot(np.cross(R1[1, :], R1[2, :]), v1))  # is 1.0 if cam1 is left and cam2 is right, -1.0 otherwise
    assert np.abs(f) > 0.1, "Sign should not be zero... ever."
    v1 *= f

    # new y axes (orthogonal to old z and new x)
    v2 = np.cross(R1[2, :], v1)
    v3 = np.cross(v1, v2)

    # catch corner case where the cameras share an optical center
    if np.sqrt(np.sum(np.square(v1))) < 1e-6:
        raise Exception("Cameras share optical center.")

    else:

        def _normed(x):
            return x / np.sqrt(np.sum(np.square(x)))

        # new extrinsics
        R = np.stack([_normed(v1), _normed(v2), _normed(v3)], 0)

    # new intrinsics
    K_new1 = K1.copy()
    K_new1[0, 1] = 0.0  # remove skew
    K_new1[:2, 2] += image_shift

    # new projection matrices
    M1_new = np.concatenate([R, -np.expand_dims(np.matmul(R, c1), 1)], 1)
    P1_new = np.matmul(K_new1, M1_new)

    M2_new = np.concatenate([R, -np.expand_dims(np.matmul(R, c2), 1)], 1)
    P2_new = np.matmul(K_new1, M2_new)

    # rectifying image transformation
    T1 = np.matmul(P1_new[:3, :3], np.linalg.inv(P1[:3, :3]))
    T2 = np.matmul(P2_new[:3, :3], np.linalg.inv(P2[:3, :3]))

    # make 4x again
    M1_new = np.concatenate([M1_new, np.array([[0.0, 0.0, 0.0, 1.0]])], 0)
    M2_new = np.concatenate([M2_new, np.array([[0.0, 0.0, 0.0, 1.0]])], 0)

    return T1, T2, K_new1, M1_new, M2_new


def rectify_image_pair(image1, K1, M1, image2, K2, M2, dist1=None, dist2=None, verbose=False):
    """Given two images and their calibration this applies Fusiello's method to rectify the images.
    http://www.diegm.uniud.it/fusiello/demo/rect/   Computer Vision Toolkit m-files/rectifyP.m
    Inputs:
        images as np.uint8
        K1, K2 as 3x3 matrices
        M1, M2 as 4x4 matrices describing the trafo from world to cam.
        dist1, dist2 as 1x5 matrices containing OpenCV distortion parameters.
                If none are given assumes image1 and image2 are already undistorted.
    Returns:
        img_rect1_left, K_left, M_left, img_rect_right, K_right, M_right
    """
    # figure relative trafos
    M_two2one = np.matmul(M1, np.linalg.inv(M2))  # cam1 ... trafo(world to cam1) * trafo(cam2 to world) ... cam2
    t_two2one = M_two2one[
        :3, 3
    ]  # if you take 0 (in cam2 coords) and trafo it this tells you where 2 is from 1's perspective
    M_one2two = np.linalg.inv(M_two2one)  # trafo from cam1 to cam2
    t_one2two = M_one2two[:3, 3]  # where cam1 is from 2's view

    # detect if it is even possible to rectify (if the cameras mostly moved in z direction)
    see1 = can_see(
        t_two2one, np.eye(4), K1, dist1, image1.shape[:2]
    )  # check if cam1 can see the optical center of cam2
    see2 = can_see(
        t_one2two, np.eye(4), K2, dist2, image2.shape[:2]
    )  # check if cam2 can see the optical center of cam1

    if see1 or see2:
        raise Exception("Cameras can see each others centrums. Impossible to rectify.")

    # rotate images such that we work in an left/right setting
    if np.abs(t_two2one[1]) > np.abs(t_two2one[0]):
        # cams are translated along y axis
        if verbose:
            print("Detected up/down setting")

        image1, K1, M1, image2, K2, M2 = rotate_images_90deg(image1, K1, M1, image2, K2, M2)

        # figure relative trafos (again)
        M_two2one = np.matmul(M1, np.linalg.inv(M2))  # cam1 ... trafo(world to cam1) * trafo(cam2 to world) ... cam2
        t_two2one = M_two2one[
            :3, 3
        ]  # if you take 0 (in cam2 coords) and trafo it this tells you where 2 is from 1's perspective
        # M_one2two = np.linalg.inv(M_two2one)  # trafo from cam1 to cam2
        # t_one2two = M_one2two[:3, 3]  # where cam1 is from 2's view

    else:
        if verbose:
            print("Detected left/right setting")

    if t_two2one[0] < 0:  # check where 2 is from 1's view
        if verbose:
            print("Swapping cams because they seem to be right/left instead of left/right")
        # cam 2 is to the 'left' of cam 1
        image1, K1, dist1, M1, image2, K2, dist2, M2 = image2, K2, dist2, M2, image1, K1, dist1, M1  # swap cams

    # rectify without centering
    T1, T2, _, _, _ = _calc_rect_homog_lr(K1, M1, K2, M2)

    # put pp into image center
    pp = np.array([K1[0, 2], K1[1, 2], 1.0])  # principal point
    pp_t = np.matmul(T1, np.expand_dims(pp, 1)).squeeze()  # how pp moved due to rectifying homography
    pp_t = pp_t[:2] / pp_t[-1]
    c = np.array([image1.shape[1] / 2.0, image1.shape[0] / 2.0])  # image center
    offset = c - pp_t  # offset such that in image1 the pp is in the image center

    # undistort images
    if dist1 is not None:
        image1 = cv2.undistort(image1, K1, dist1)
    if dist2 is not None:
        image2 = cv2.undistort(image2, K2, dist2)

    # rectify with centering
    T1, T2, K, M1, M2 = _calc_rect_homog_lr(K1, M1, K2, M2, offset)

    # warp images with the found homography
    img_shape = image1.shape[:2]
    img_shape = img_shape[::-1]
    img_rect1 = cv2.warpPerspective(image1, T1, img_shape)
    img_rect2 = cv2.warpPerspective(image2, T2, img_shape)

    return img_rect1, img_rect2, K, M1, M2


def _skew(vec):
    """skew-symmetric cross-product matrix of vec."""
    A = np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]])
    return A


def _get_aligning_rotation(vec1, vec2):
    """
    Returns a rotation such that vec2 = R * vec1
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677
    """
    vec1 = np.reshape(vec1, [3])
    vec2 = np.reshape(vec2, [3])
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)

    if np.abs(s) < 1e-6:
        return np.eye(3)

    if np.abs(c + 1.0) < 1e-3:
        assert 0, "CORNER CASE"

    v_skew = _skew(v)
    v_skew_prod = np.matmul(v_skew, v_skew)
    R = np.eye(3) + v_skew + v_skew_prod * (1.0 - c) / (s * s)
    return R


def center_pp_warp(image, cam_intrinsic, points3d=None, return_warped=True, return_homography=False, borderValue=0):
    """Puts the principal point into the image center.

    Calculates a virtual camera that is a rotated version of the original camera.
    The rotation is chosen such that the virtual cameras' optical axis is 'pointing' into the 3D direction
    defined by the center of the current (cropped) image. Given the two cameras a homography is estimated
    that allows for warping the image content from the real to the virtual camera.

    IMPORTANT: points3d MUST be coordinates in the cameras coordinate frame!

    HOW TO PROJECT:
    1) Keep crop coordinates an correct perspective in 2D
    uv = cl.project(cl.trafo_coords(xyz, M), K_crop)
    uv = cl.trafo_coords(uv, H)

    2) Change 3D points to accomodate the perspective change
    M = np.matmul(R.T, M)
    uv = cl.project(cl.trafo_coords(xyz, M), K_hat)
    """
    # 1. Calculate new optical axis
    w, h = image.shape[1], image.shape[0]
    pp = np.array([[w / 2.0, h / 2.0]])
    axis_new = backproject(pp, np.ones_like(pp[:, :1]), cam_intrinsic)

    # 2. Calculate world rotation
    axis_old = np.array([[0.0, 0.0, 1.0]])
    R = _get_aligning_rotation(axis_old, axis_new)
    # tmp = np.matmul(axis_old, R.T)
    # print('axis_new', axis_new)  # they are the same
    # print('tmp', tmp)

    # 3. New camera intrinsic (which we would like to have)
    cam_intrinsic_warp = np.array(
        [[cam_intrinsic[0, 0], 0.0, w / 2.0], [0.0, cam_intrinsic[1, 1], h / 2.0], [0.0, 0.0, 1.0]]
    )

    # 4. Create some control points for homography estimation
    points3d_dummy = np.array(
        [
            [1.0, 0.0, 1.0],  # doesn matter which points we use. the math should hold anyways
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points3d_dummy_new = np.matmul(points3d_dummy, R)

    # 5. Project control points into both cameras
    points2d_dummy_new = project(points3d_dummy_new, cam_intrinsic_warp)
    points2d_dummy = project(points3d_dummy, cam_intrinsic)

    # 6. Estimate homography from the corresponding control points
    H, _ = cv2.findHomography(
        np.reshape(points2d_dummy, [4, 1, 2]).astype(np.float32),
        np.reshape(points2d_dummy_new, [4, 1, 2]).astype(np.float32),
        method=0,
    )
    # tmp = _from_hom(np.matmul(_to_hom(points2d_dummy), H.T))
    # print('points2d_dummy_new', points2d_dummy_new)  # they are the same
    # print('tmp', tmp)

    outputs = [R, cam_intrinsic_warp]
    # 7. Warp image according to homography
    if return_warped:
        image_warp = cv2.warpPerspective(image, H, (h, w), borderValue=borderValue)
        outputs.append(image_warp)

    # 8. Transform world points accordingly if wanted
    if points3d is not None:
        # calculate new world points if they were given
        points3d = np.matmul(points3d, R)
        outputs.append(points3d)

    if return_homography:
        outputs.append(H)

    return outputs


def resize(image, K, target_shape_hw):
    """Resizes image to target_shape and adapts intrinsics accordingly."""
    ratio = np.array(target_shape_hw, dtype=np.float32) / np.array(image.shape[:2], dtype=np.float32)
    image = cv2.resize(image, target_shape_hw[::-1])
    K = K.copy()
    K[0, :] *= ratio[1]
    K[1, :] *= ratio[0]
    return image, K


def estimate_root_xyz(kp_uv, xyz_rel, K):
    """Estimates the root keypoint from a given root normalized (but correctly scaled)
    3D estimation and the camera intrinsic matrix.

    kp_uv, Nx2, projected keypoints.
    xyz_rel, Nx3, root relative 3D keypoints.
    K, 3x3, Intrinsic camera calibration.
    """
    # get more shorter variable names
    f_u = K[0, 0]
    f_v = K[1, 1]
    pp_u = K[0, 2]
    pp_v = K[1, 2]

    # assemble measurement matrices
    N_obs = kp_uv.shape[0]
    A = np.zeros((2 * N_obs, 3))
    b = np.zeros((2 * N_obs,))
    A[:N_obs, 0] = -1
    A[:N_obs, 2] = (kp_uv[:, 0] - pp_u) / f_u
    A[N_obs:, 1] = -1
    A[N_obs:, 2] = (kp_uv[:, 1] - pp_v) / f_v

    b[:N_obs] = xyz_rel[:, 0] + xyz_rel[:, 2] * (pp_u - kp_uv[:, 0]) / f_u
    b[N_obs:] = xyz_rel[:, 1] + xyz_rel[:, 2] * (pp_v - kp_uv[:, 1]) / f_v

    # solve least squares problem
    root_pred, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return root_pred


def project_ortho(xy, scales, pp):
    """Orthographic projection."""
    uv = scales * xy[:, :2] + pp
    return uv


def estimate_root_xy_ortho(kp_uv, xyz_rel, pp):
    """Estimates the root keypoint from a given root normalized (but correctly scaled)
    3D estimation assuming an orthographic projection.

    Assumes an orthographic camera.

    kp_uv, Nx2, projected keypoints.
    xyz_rel, Nx3, root relative 3D keypoints.
    pp, 2, principal points
    """

    # assemble measurement matrices
    N_obs = kp_uv.shape[0]
    A = np.zeros((2 * N_obs, 3))
    b = np.zeros((2 * N_obs,))
    A[:N_obs, 0] = xyz_rel[:, 0]
    A[:N_obs, 1] = 1.0
    A[N_obs:, 0] = xyz_rel[:, 1]
    A[N_obs:, 2] = 1.0

    b[:N_obs] = kp_uv[:, 0] - pp[0]
    b[N_obs:] = kp_uv[:, 1] - pp[1]

    # solve least squares problem
    pred, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale = pred[0]
    x_root = pred[1] / scale
    y_root = pred[2] / scale

    return scale, np.expand_dims(np.stack([x_root, y_root]), 0)


def to_pcl(rgb, depth, K, dist, eps=1e-3):
    """
    Calculate a point cloud from rgb image, depth map and intrinsics.
    """
    mask = depth > eps
    z_coords = depth[mask].flatten()
    if rgb is not None:
        rgb_vals = rgb[mask]

    H, W = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij")
    H, W = H[mask].flatten().astype(np.float32), W[mask].flatten().astype(np.float32)
    uv_coords = np.stack([W, H], -1)

    xyz_coords = backproject(uv_coords, z_coords, K, dist)
    if rgb is not None:
        return xyz_coords, rgb_vals
    else:
        return xyz_coords


def depth_to_frame(depth_map, K_d, dist_d, M_d2t, K_t, dist_t, img_shape_t, mode=None):
    """Warps a given depth map into another target frame 't'."""
    # transform depth map into point cloud
    xyz = to_pcl(None, depth_map, K_d, dist_d)

    # trafo pcl into other camera
    xyz_t = trafo_coords(xyz, M_d2t)

    # project into cam
    uv_t = project(xyz_t, K_t, dist_t)

    # round to pixels (a proper library probably has better ways to deal with it)
    inds = np.stack([uv_t[:, 1], uv_t[:, 0]], -1)
    inds = np.round(inds).astype(np.int32)
    valid = np.logical_and(
        np.logical_and(0 <= inds[:, 0], inds[:, 0] < img_shape_t[0]),
        np.logical_and(0 <= inds[:, 1], inds[:, 1] < img_shape_t[1]),
    )

    # mask out invalid values (outside target image bounds)
    inds = inds[valid]
    depths = xyz_t[:, 2]
    depths = depths[valid]

    if mode is None:
        from scipy import interpolate

        grid_x, grid_y = np.mgrid[0 : img_shape_t[0], 0 : img_shape_t[1]]
        depth_t = interpolate.griddata(inds, depths, (grid_x, grid_y), method="linear", fill_value=0.0)

    elif mode == "sparse":
        # create image from points
        depth_t = np.ones(img_shape_t, dtype=np.float32) * 1e8
        for (i, j), d in zip(inds, depths):  # this loop will be painfully slow
            depth_t[i, j] = np.minimum(depth_t[i, j], d)  # do depth buffering

        # change invalid value
        depth_t[depth_t > 1e7] = 0.0
    else:
        raise NotImplementedError("Unknown interpolation mode: %s" % mode)
    return depth_t
