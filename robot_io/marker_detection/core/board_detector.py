import cv2

from robot_io.marker_detection.apriltag_detection.AprilTagDetectorBatch import *
from robot_io.marker_detection.utils.general_util import json_load


class BoardDetector(object):
    """Detects a predefined Apriltag-based calibration board in images.
    Also knows where all its landmarks lie in 3D.
    """

    def __init__(self, marker_def_file, num_parallel_jobs=10, downsampling=1):

        # load marker info from file
        marker_def = json_load(marker_def_file)
        self.marker_dim = (marker_def["n_x"], marker_def["n_y"])
        marker_type = marker_def["family"]
        assert marker_type in ["36h11", "16h5"], "Marker family not implemented."
        black_border = int(marker_def["border"])
        self.tag_size = marker_def["tsize"]  # marker size in m
        self.tag_spacing = marker_def["tspace"]  # space between tags in m
        self.offset = marker_def["offset"]  # offset between front and back in m
        self.double = marker_def["double"]  # is it a double sided tag?
        self.id_start = marker_def.get("id_start", 0)  # some marker dont start at id 0

        self.tag_detector_batch = PyRunAprilDetectorBatch(
            marker_type, black_border, num_parallel_jobs, 1.0 / downsampling
        )
        self.object_points = self.get_april_tag_points()

    def _front2back(self, points_front, shift):
        """Given the 3d model points of the front side it calculates the location of the points on the back side."""
        points_back = np.reshape(points_front.copy(), [self.marker_dim[0], self.marker_dim[1], 4, 3])
        points_back = points_back[:, ::-1, :, :]  # transpose along x axis
        tmp = points_back.copy()
        points_back[:, :, 0, :] = tmp[:, :, 1, :]
        points_back[:, :, 1, :] = tmp[:, :, 0, :]
        points_back[:, :, 2, :] = tmp[:, :, 3, :]
        points_back[:, :, 3, :] = tmp[:, :, 2, :]
        points_back = np.reshape(points_back, [-1, 3]) + shift
        return points_back

    def _get_corners_of_tag(self):
        """Returns the corner points of a single AprilTag in the right order (CCW)."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0]) * self.tag_size
        p2 = np.array([1.0, 1.0, 0.0]) * self.tag_size
        p3 = np.array([0.0, 1.0, 0.0]) * self.tag_size
        return np.stack([p0, p1, p2, p3])

    def get_april_tag_points(self):
        """Returns the corner points wrt. of a CalibBoard wrt to an relative 3D coordinate frame."""
        p_tag_corners = self._get_corners_of_tag()

        # calculate front points
        obj_points = list()
        for yid in range(self.marker_dim[0]):
            for xid in range(self.marker_dim[1]):
                # calculate location of starting point of each tag
                x = xid * (self.tag_size + self.tag_spacing)
                y = yid * (self.tag_size + self.tag_spacing)
                p_start = np.array([[x, y, 0.0]])

                obj_points.append(p_start + p_tag_corners)
        obj_points = np.concatenate(obj_points)

        if self.double:
            # calculate back points
            obj_points_back = self._front2back(obj_points, self.offset)

            # join them
            obj_points = np.concatenate([obj_points, obj_points_back], 0)

        return obj_points

    def get_matching_objectpoints(self, point2d_ids, point2d_uv=None):
        """Gives you 3D coordinates of the board which match the 2D points."""

        # mask invalid ones
        point2d_ids = point2d_ids - 4 * self.id_start
        m = np.logical_and(point2d_ids >= 0, point2d_ids < self.object_points.shape[0])
        point2d_ids = point2d_ids[m]

        # select points that were detected
        object_points_det = self.object_points[point2d_ids, :]

        if point2d_uv is None:
            return object_points_det

        point2d_uv = point2d_uv[m]
        return object_points_det, point2d_uv

    def process_image_batch(self, image_file_list):
        """Detects points on a given list of strings (image paths) and returns a list of detections."""
        det_list = self.tag_detector_batch.processImageBatch(image_file_list)

        point_coords_frames = list()
        point_ids_frames = list()
        for fid, det_f in enumerate(det_list):
            point_coords = list()
            point_ids = list()
            for det in det_f:
                point_coords.append(np.array(det.points))
                point_ids.append(4 * det.id)
                point_ids.append(4 * det.id + 1)
                point_ids.append(4 * det.id + 2)
                point_ids.append(4 * det.id + 3)

            # turn into np array
            if len(point_coords) > 0:
                point_coords = np.concatenate(point_coords)
            else:
                point_coords = np.zeros((0, 2))

            point_coords_frames.append(point_coords)
            point_ids_frames.append(point_ids)

        return point_coords_frames, point_ids_frames

    def process_image(self, image_file):
        det_f = self.tag_detector_batch.processImage(image_file)
        point_coords = list()
        point_ids = list()
        for det in det_f:
            point_coords.append(np.array(det.points))
            point_ids.append(4 * det.id)
            point_ids.append(4 * det.id + 1)
            point_ids.append(4 * det.id + 2)
            point_ids.append(4 * det.id + 3)

        # turn into np array
        if len(point_coords) > 0:
            point_coords = np.concatenate(point_coords)
        else:
            point_coords = np.zeros((0, 2))
        return point_coords, point_ids

    def process_image_m(self, image):
        det_f = self.tag_detector_batch.processImageM(image)
        point_coords = list()
        point_ids = list()
        for det in det_f:
            point_coords.append(np.array(det.points))
            point_ids.append(4 * det.id)
            point_ids.append(4 * det.id + 1)
            point_ids.append(4 * det.id + 2)
            point_ids.append(4 * det.id + 3)

        # turn into np array
        if len(point_coords) > 0:
            point_coords = np.concatenate(point_coords)
        else:
            point_coords = np.zeros((0, 2))
        return point_coords, point_ids

    def draw_board(self, image, points, point_ids, linewidth=2, cross_out=False, tagId=None, show=True, block=True):
        # inpaint the image
        for i in range(0, points.shape[0], 4):
            self.draw_tag(image, points[i : (i + 4), :], tagId=point_ids[i] / 4)

        # optionally show
        if show:
            image = cv2.resize(image, (480, 240))
            cv2.namedWindow("board_det")
            cv2.imshow("board_det", image)

            time = 10
            if block:
                time = -1
            cv2.waitKey(time)

    @staticmethod
    def draw_tag(img, point_list, linewidth=2, cross_out=False, tagId=None):
        c = (255, 255, 0)
        g = (0, 255, 0)
        lines = [(0, 1, c), (1, 2, g), (2, 3, g), (3, 0, g)]  # bottom, right, top, left (goes around CCW)
        for l in lines:
            c0 = (int(point_list[l[0]][0]), int(point_list[l[0]][1]))
            c1 = (int(point_list[l[1]][0]), int(point_list[l[1]][1]))
            cv2.line(img, c0, c1, color=l[2], thickness=8)

    @staticmethod
    def show_tag(ax, point_list, linewidth=2, cross_out=False, tagId=None):
        lines = [(0, 1, "c"), (1, 2, "g"), (2, 3, "g"), (3, 0, "g")]  # bottom, right, top, left (goes around CCW)
        for l in lines:
            ax.plot(
                [point_list[l[0]][0], point_list[l[1]][0]],
                [point_list[l[0]][1], point_list[l[1]][1]],
                l[2],
                linewidth=linewidth,
            )

        if cross_out:
            ax.plot(
                [point_list[0][0], point_list[2][0]], [point_list[0][1], point_list[2][1]], "r", linewidth=linewidth
            )
            ax.plot(
                [point_list[1][0], point_list[3][0]], [point_list[1][1], point_list[3][1]], "r", linewidth=linewidth
            )

        if tagId is not None:
            middle = np.mean(np.array(point_list), 0)
            ax.text(
                middle[0],
                middle[1],
                "%d" % tagId,
                fontsize=25,
                color="r",
                horizontalalignment="center",
                verticalalignment="center",
            )
