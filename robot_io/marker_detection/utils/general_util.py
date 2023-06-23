from __future__ import print_function, unicode_literals

from collections import namedtuple
import datetime
import json
import os
import re
import time

import commentjson
import numpy as np
import PIL
from scipy.ndimage import find_objects, label
from scipy.ndimage.filters import maximum_filter
from scipy.spatial import procrustes

CamCalib = namedtuple("CamCalib", ["K", "dist", "M"])


def read_disp_file(name):
    f = open(name, "rb")

    if (f.readline().decode("utf-8")) != "float\n":
        raise Exception("float file %s did not contain <float> keyword" % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def read_flo_file(name):
    assert name.endswith(".flo"), "Name should end with flo"

    f = open(name, "rb")

    header = f.read(4)
    if header.decode("utf-8") != "PIEH":
        raise Exception("Flow file header does not contain PIEH")

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def try_to_match(string, wildcard, max_id=128):
    for i in range(max_id):
        if wildcard % i in string:
            return i
    return None


def parse_file_name(path, run_wc, cam_wc=None):
    assert os.path.exists(path), "Path not found!"
    assert os.path.isfile(path), "Path is not a file!"
    base_path = os.path.dirname(path)
    file_name, file_ext = os.path.splitext(os.path.basename(path))
    run_id = try_to_match(file_name, run_wc)
    if cam_wc is None:
        return base_path, run_id

    cam_id = try_to_match(file_name, cam_wc)
    return base_path, run_id, cam_id


def my_mkdir(path, is_file):
    if is_file:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def sample_uniform(l, n):
    """Yield n uniformly sampled items from the list."""
    if n == 1:
        yield l[len(l) // 2]

    else:
        step = int((len(l) - 1) / float(n - 1))
        for i in range(0, len(l), step):
            yield l[i]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def split(list_obj, num_parts):
    k, m = divmod(len(list_obj), num_parts)
    return (list_obj[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in xrange(num_parts))


def search_files(base_path, pattern, depth=0, verbose=False, max_depth=5):
    base_path = os.path.abspath(base_path)

    if depth >= max_depth:
        return []
    s = "".join(["\t" for _ in range(depth)])
    if verbose:
        print("%s Searching: " % s, base_path)
    json_files = list()
    for dirpath, dirnames, filenames in os.walk(base_path):
        if verbose:
            print("%s Found: %d dirs and %d files in %s" % (s, len(dirnames), len(filenames), dirpath))
        # search depth first
        for d in dirnames:
            if verbose:
                print("%s Entering dir: " % s, d)
            json_files.extend(search_files(os.path.join(dirpath, d), pattern, depth + 1, verbose, max_depth))

        if verbose:
            print("%s Found %d file(s) here" % (s, len(filenames)))

        for f in filenames:
            if verbose:
                print('Checking file "%s"' % f)
            if pattern in f:
                if verbose:
                    print("%s Found a json: %s" % (s, os.path.join(base_path, dirpath, f)))
                json_files.append(os.path.join(base_path, dirpath, f))
        return json_files


def listify(x):
    if not type(x) == list:
        return [x]
    return x


def _tryint(s):
    try:
        return int(s)
    except:
        return s


def _alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [_tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=_alphanum_key)


def flatten_list(this):
    """Turns a nested list structure into one long list."""
    if type(this) == list:
        # if its a list flatten each item
        flattened = list()
        for item in this:
            if type(item) == list:
                flattened.extend(flatten_list(item))
            else:
                flattened.append(flatten_list(item))
        return flattened
    else:
        return this


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if callable(obj):
            return None
        return json.JSONEncoder.default(self, obj)


def json_dump(file_name, data, pretty_format=False, overwrite=True, verbose=False):
    msg = "File does exists and should not be overwritten: %s" % file_name
    assert not os.path.exists(file_name) or overwrite, msg

    with open(file_name, "w") as fo:
        if pretty_format:
            json.dump(data, fo, cls=NumpyEncoder, sort_keys=True, indent=4)
        else:
            json.dump(data, fo, cls=NumpyEncoder)

    if verbose:
        print("Dumped %d entries to file %s" % (len(data), file_name))


def json_load(file_name, verbose=False):
    if file_name.endswith(".cfg.json"):
        # if the file contains the intermediate .cfg.json ending we use the library that can use comments.
        with open(file_name, "r") as fi:
            data = commentjson.load(fi)
    else:
        with open(file_name, "r") as fi:
            data = json.load(fi)

    if verbose:
        print("Loaded %d items from %s" % (len(data), file_name))
    return data


class Timer:
    def __init__(self, text=None, show=True):
        if text is None:
            text = "time passed"

        self.text = text
        self.show = show

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        interval = time.time() - self.start

        if self.show:
            msg = ""
            if self.text is not None:
                msg += "%s " % self.text
            msg += "%.2f sec" % interval
            print(msg)


def preprocess_image(
    img_raw,
    crop=None,
    do_mean_subtraction=True,
    over_sampling=2.2,
    symmetric=False,
    resize=True,
    target_size=368,
    borderValue=None,
    raise_exp=True,
):
    """Reads an input image and preprocesses it appropriately to be fed into the network."""
    if borderValue is None:
        borderValue = 0.0
        if do_mean_subtraction:
            borderValue = 127.5

    expanded = False
    if len(img_raw.shape) == 2:
        img_raw = np.expand_dims(img_raw, -1)
        expanded = True

    import cv2

    # resize, RGB -> BGR and subtract mean
    img_raw = img_raw.copy()
    offset = np.array([0.0, 0.0])
    if crop is not None:
        # make sure we dont exceed image dimensions
        shape = np.array(img_raw.shape[:2])
        start = crop[0, :]
        end = crop[1, :]
        center = 0.5 * (start + end)
        size = 0.5 * (end - start)
        if symmetric:
            size = np.ones_like(size) * np.max(size)
        size *= over_sampling  # sample larger

        # round crop dimensions to full pixels
        crop_size = (2 * size).round().astype(np.int32)

        # create crop image
        img_crop = borderValue * np.ones(
            (crop_size[0], crop_size[1], 3), dtype=np.float32
        )  # after mean subtraction 127.5 will be zero

        # figure out where we would like to crop (can exceed image dimensions)
        start_t = (center - size).round().astype(np.int32)
        end_t = start_t + crop_size

        # check if there is actually anything to be cropped (sometimes crop is completely out of the image).
        do_crop = True

        # sanity check the crop values (sometime the crop is completely outside the image)
        if np.any(np.logical_or(end_t < 0, start_t > shape - 1)):
            print("WARNING: Crop is completely outside image bounds!")
            do_crop = False
            if raise_exp:
                raise Exception

        # check image boundaries: Where can we crop?
        start = np.maximum(start_t, 0)
        end = np.minimum(end_t, shape - 1)

        # check discrepancy
        crop_start = start - start_t
        crop_end = crop_size - (end_t - end)

        if do_crop:
            img_crop[crop_start[0] : crop_end[0], crop_start[1] : crop_end[1], :] = img_raw[
                start[0] : end[0], start[1] : end[1], :
            ]

        img_raw = img_crop
        offset = start - crop_start

    scale = np.array([1.0, 1.0])
    if resize:
        scale = img_raw.shape[:2] / np.array([target_size, target_size], dtype=np.float32)
        img_raw = cv2.resize(img_raw, (target_size, target_size))

    img = np.array(img_raw, dtype=np.float32)
    img = img[:, :, ::-1]  # rgb --> bgr

    if do_mean_subtraction:
        img = img / 256.0 - 0.5  # MEAN

    if expanded:
        img = img.mean(-1)
        img_raw = img_raw.mean(-1)

    return img, scale, offset, img_raw

    # # resize, RGB -> BGR and subtract mean
    # img_raw = img_raw.copy()
    # offset = np.array([0.0, 0.0])
    # if crop is not None:
    #     # make sure we dont exceed image dimensions
    #     shape = np.array(img_raw.shape[:2])
    #     start = crop[0, :]
    #     end = crop[1, :]
    #     center = 0.5 * (start + end)
    #     size = 0.5 * (end - start)
    #     size *= 2.2  # sample larger
    #     start = center - size
    #     end = center + size
    #
    #     # check image boundaries
    #     start = np.maximum(start, 0).round().astype(np.int32)
    #     end = np.minimum(end, shape - 1).round().astype(np.int32)
    #
    #     # get crop
    #     img_raw = img_raw[start[0]:end[0], start[1]:end[1], :]
    #     offset = start
    #
    # scale = img_raw.shape[:2] / np.array([368.0, 368.0])
    # import scipy.misc
    # img_raw = scipy.misc.imresize(img_raw, (368, 368))
    # img = np.array(img_raw, dtype=np.float32)
    # img = img[:, :, ::-1]  # rgb --> bgr
    #
    # img = img / 255.0 - 0.5  # MEAN
    # return img, scale, offset, img_raw


def compensate_crop_K(K, scale, offset):
    K = K.copy()
    A = np.array(
        [[1.0 / scale[1], 0.0, -offset[1] / scale[1]], [0.0, 1.0 / scale[0], -offset[0] / scale[0]], [0.0, 0.0, 1.0]]
    )
    return np.matmul(A, K)


def compensate_crop_coords(points2d, scale, offset):
    """From original to crop."""
    points2d = points2d.copy()
    points2d[:, 0] -= offset[1]
    points2d[:, 1] -= offset[0]
    points2d[:, 0] /= scale[1]
    points2d[:, 1] /= scale[0]
    return points2d


def compensate_crop_coords_inv(points2d, scale, offset):
    """From crop to original."""
    points2d = points2d.copy()
    points2d[:, 0] *= scale[1]
    points2d[:, 1] *= scale[0]
    points2d[:, 0] += offset[1]
    points2d[:, 1] += offset[0]
    return points2d


def unsigmoid(array):
    return -np.log(np.maximum(1.0 / (array + 1e-6) - 1.0, 1e-6))


def sigmoid(array):
    # m = -array > np.log(np.finfo(type(value)).max)
    return 1.0 / (1.0 + np.exp(-array))


def softmax(array, axes):
    tmp = array.copy()
    tmp = np.exp(tmp)
    return tmp / np.sum(tmp, axes, keepdims=True)


def softargmax_np(scoremap):
    """Returns the coordinates from scoremap."""
    expanded = False
    if len(scoremap.shape) == 3:
        expanded = True
        scoremap = np.expand_dims(scoremap, 0)

    s = scoremap.shape
    rng_h = np.arange(s[1])
    rng_w = np.arange(s[2])
    H, W = np.meshgrid(rng_h, rng_w, indexing="ij")
    H, W = np.expand_dims(np.expand_dims(H, 0), -1), np.expand_dims(np.expand_dims(W, 0), -1)
    score_distribution = softmax(scoremap, (1, 2))
    h, w = np.sum(score_distribution * H, (1, 2)), np.sum(score_distribution * W, (1, 2))
    coords_hw = np.stack([h, w], 2)
    conf = np.max(sigmoid(scoremap), (1, 2))
    if expanded:
        coords_hw = np.squeeze(coords_hw, 0)
        conf = np.squeeze(conf, 0)
    return coords_hw, conf


def detect_keypoints_softargmax(scoremaps):
    """Performs detection per scoremap for the hands keypoints."""
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = np.zeros((s[2], 2))
    conf = list()
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
        conf.append(scoremaps[v, u, i])
    return keypoint_coords, np.stack(conf)


def detect_keypoints(scoremaps):
    """Performs detection per scoremap for the hands keypoints."""
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = np.zeros((s[2], 2))
    conf = list()
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
        conf.append(scoremaps[v, u, i])
    return keypoint_coords, np.stack(conf)


def detect_multiple_keypoints(scoremaps, neighborhood=5, min_score=0.1):
    """Performs detection per scoremap for the hands keypoints."""
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    # 1. Maximum filter: Some kind of Non-maximum suppression
    scorevol_max = maximum_filter(scoremaps, size=neighborhood, mode="constant", cval=0.0)
    maxima = scorevol_max == scoremaps
    maxima[scoremaps < min_score] = False

    kp2d, conf = list(), list()
    for i in range(s[2]):
        # 2. Find distinct maxima
        labeled, num_objects = label(maxima[:, :, i])
        slices = find_objects(labeled)

        if num_objects == 0:
            kp2d.append(None)
            continue

        # 3. Create matrix of found maxima with their location
        objects = np.zeros((num_objects, 3), dtype=np.float32)
        for oid, (dx, dy) in enumerate(slices):
            objects[oid, :2] = [(dx.start + dx.stop - 1) / 2, (dy.start + dy.stop - 1) / 2]
            X, Y = np.round(objects[oid, :2]).astype(np.int32)
            objects[oid, 2] = scoremaps[X, Y, i]
        kp2d.append(objects)

    return kp2d


def load_calib_data(cam_calib_file, return_cam2world):
    """Loads camera calibration data from a json file."""
    # read calibration
    with open(cam_calib_file, "r") as fi:
        calib = json.load(fi)

    # convert into useful form
    cam_intrinsic, cam_dist, cam_extrinsic = dict(), dict(), dict()
    for camName in calib["K"].keys():
        cam_intrinsic[camName] = np.array(calib["K"][camName])
        cam_dist[camName] = np.array(calib["dist"][camName])
        if return_cam2world:
            # trafo that transforms camera into world coordinates
            cam_extrinsic[camName] = np.array(calib["M"][camName])
        else:
            # trafo that transforms world into camera coordinates
            cam_extrinsic[camName] = np.linalg.inv(np.array(calib["M"][camName]))

    out = dict()
    for k in cam_intrinsic.keys():
        # out[k] = CamCalib(K=cam_intrinsic[k], dist=cam_dist[k], M=cam_extrinsic[k])
        out[k] = {"K": cam_intrinsic[k], "dist": cam_dist[k], "M": cam_extrinsic[k]}
    return out


def normalize_intrinsics(K_dict, img_shapes):
    """Makes intrinsic normalized wrt image scale."""

    def _mod(K, img_shape):
        K = K.copy()
        K[0, :] /= float(img_shape[1])
        K[1, :] /= float(img_shape[0])
        return K

    K_new = dict()
    for cam_name in K_dict.keys():
        if type(img_shapes) == dict:
            shape = img_shapes[cam_name]
        else:
            shape = img_shapes
        K_new[cam_name] = _mod(K_dict[cam_name], shape)
    return K_new


def get_intrinsics_in_crop(K_dict, scale_offset_dict, crop_is_upscaled):
    """Makes the intrinsic map into the crop."""

    def _mod(K, scale, offset):
        if not crop_is_upscaled:
            scale *= 8.0
        K = K.copy()
        A = np.array(
            [
                [1.0 / scale[1], 0.0, -offset[1] / scale[1]],
                [0.0, 1.0 / scale[0], -offset[0] / scale[0]],
                [0.0, 0.0, 1.0],
            ]
        )
        return np.matmul(A, K)

    K_new = dict()
    for cam_name in K_dict.keys():
        scale, offset = scale_offset_dict[cam_name]
        K_new[cam_name] = _mod(K_dict[cam_name], scale, offset)
    return K_new


def align_procrustes(xyz_gt, xyz_pred):
    t = xyz_gt.mean(1, keepdims=True)
    xyz_gt -= xyz_gt.mean(1, keepdims=True)
    xyz_pred -= xyz_pred.mean(1, keepdims=True)

    xyz_gt_n, xyz_pred_n = list(), list()
    for bid in range(xyz_gt.shape[0]):
        t1, t2, _ = procrustes(xyz_gt[bid], xyz_pred[bid])
        xyz_gt_n.append(t1)
        xyz_pred_n.append(t2)
    xyz_gt_n, xyz_pred_n = np.stack(xyz_gt_n), np.stack(xyz_pred_n)

    scale = np.mean(np.mean(xyz_gt[:, 1:, :] / xyz_gt_n[:, 1:, :], 1, keepdims=True), 2, keepdims=True)
    xyz_pred_aligned = scale * xyz_pred_n
    return xyz_pred_aligned


def rigid_transform_3D(A, B):
    """Calculates R, t to align point sets A and B: B' = R * A.T + t
    A is NxD
    B is NxD
    """
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.matmul(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A.T) + centroid_B.T

    return R, np.expand_dims(t, 0)


def _skew(vec):
    """skew-symmetric cross-product matrix of vec."""
    A = np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]])
    return A


def get_aligning_rotation(vec1, vec2):
    """
    Returns a rotation such that vec2 = R * vec1
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677
    """
    vec1 = np.reshape(vec1.copy(), [3])
    vec2 = np.reshape(vec2.copy(), [3])
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)

    if np.abs(s) < 1e-6:
        return np.eye(3)

    if np.abs(c + 1.0) < 1e-3:
        # a and b point into exactly opposite directions
        i = np.argmax(np.abs(vec1))  # find maximal direction
        i = (i + 1) % 3  # chose any direction that is not maximal
        R = np.eye(3) * -1
        R[:, i] *= -1
        return R

    v_skew = _skew(v)
    v_skew_prod = np.matmul(v_skew, v_skew)
    R = np.eye(3) + v_skew + v_skew_prod * (1.0 - c) / (s * s)
    return R


def calc_iou(mask_gt, mask_pred, eps=1e-8):
    """Calculate intersection over union for binary mask images."""
    assert mask_gt.shape == mask_pred.shape, "Shape between masks mismatches."
    assert len(mask_gt.shape) == 2, "Shape should be 2D."

    # make binary
    mask_gt = mask_gt.copy() > 0
    mask_pred = mask_pred.copy() > 0

    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    return np.sum(intersection) / (np.sum(union) + eps)


class Logger:
    """Class used for printing and logging stuff.

    Provide a logfile on start-up."""

    def __init__(self, path_to_file, mode="w", print_msg=True, time_stamp_fct=None):
        self.path_to_file = path_to_file
        if (mode == "w") or (mode == "a"):
            self.mode = mode
        else:
            assert 0, "Invalid logging mode"
        self.fo = None
        self.ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        self.print_msg = print_msg
        self.time_stamp_fct = time_stamp_fct

    def __enter__(self):
        self.fo = open(self.path_to_file, self.mode)
        return self

    def log(self, text, end="\n", print_this=True):
        print_this = print_this and self.print_msg

        msg = ""
        if self.time_stamp_fct is not None:
            msg += "[%s]: " % self.time_stamp_fct()
        msg += str(text)
        if print_this:
            print(msg, end=end)
        msg += end

        try:
            msg = self.ansi_escape.sub("", msg)  # remove color code
            self.fo.write(msg)

        except OSError:
            # If it fails, reopen the file
            self.fo = open(self.path_to_file, "a")
            self.fo.write("File was opened again due to a stale file descriptor.\n")
            self.fo.write(msg)

    def flush_data(self):
        self.fo.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fo.write("File was closed.\n")
        self.fo.close()


def calc_bbox(uv, vis=None):
    if vis is None:
        vis = np.ones_like(uv[:, 0])

    uv_int = np.round(uv).astype(np.int32)
    min_v, max_v = np.min(uv_int[vis > 0.5, :], 0), np.max(uv_int[vis > 0.5, :], 0)
    bbox = np.array([[min_v[1], min_v[0]], [max_v[1], max_v[0]]])

    return bbox


def get_img_size(img_path):
    # load image size
    im = PIL.Image.open(img_path)
    width, height = im.size
    return width, height


def now_time_str():
    return datetime.datetime.now().strftime("%H-%M-%S.%f")[:-3]


def find_first_non_existant(search_path):
    for i in range(1024):
        if not os.path.exists(search_path % i):
            return search_path % i
    return None


def find_last_existant(search_path):
    last_ex = None
    for i in range(1024):
        if os.path.exists(search_path % i):
            last_ex = search_path % i
    return last_ex


class EarlyStoppingUtil(object):
    def __init__(self, thresh_steps, min_rel_improvement=0.0):
        self.thresh_steps = thresh_steps
        self.min_rel_improvement = (
            min_rel_improvement  # minimal improvement for not triggering early stopping, positive is improvement
        )

        self.best_result = None
        self.num_cond_break = 0

    def feed(self, result):
        if self.best_result is None:
            self.best_result = result
            return

        rel_improvement = (self.best_result - result) / (1e-4 + self.best_result)  # positive if it has gotten better
        if rel_improvement > self.min_rel_improvement:
            print("EarlyStoppingUtil: New best with feed %.2f, rel_impr=%.2f" % (result * 100, rel_improvement))
            # new best result
            self.best_result = result
            self.num_cond_break = 0
        else:
            print("EarlyStoppingUtil: No new best feed %.2f, rel_impr=%.2f" % (result * 100, rel_improvement))
            self.num_cond_break += 1

    def should_stop(self):
        if self.best_result is None:
            return False
        return self.num_cond_break >= self.thresh_steps
