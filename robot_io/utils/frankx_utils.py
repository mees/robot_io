from frankx import Affine

from robot_io.utils.utils import euler_to_quat, scipy_quat_to_np_quat


def to_affine(target_pos, target_orn):
    if len(target_orn) == 3:
        target_orn = euler_to_quat(target_orn)
    target_orn = scipy_quat_to_np_quat(target_orn)
    return Affine(*target_pos, target_orn.w, target_orn.x, target_orn.y, target_orn.z)
