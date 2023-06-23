from frankx import Affine, Kinematics, NullSpaceHandling
import numpy as np

from robot_io.robot_interface.panda_frankx_interface import to_affine
from robot_io.utils.utils import matrix_to_pos_orn, pos_orn_to_matrix


class IKFrankX:
    def __init__(
        self,
        nullspace_joint_id,
        nullspace_joint_value,
        ll,
        ul,
    ):
        self.null_space = NullSpaceHandling(nullspace_joint_id, nullspace_joint_value)
        self.ll = ll
        self.ul = ul
        # self.EE_T_NE = np.linalg.inv(np.array([[1, 0, 0, 0],
        #                                         [0, -1, 0, 0],
        #                                         [0, 0, -1, 0],
        #                                         [0, 0, 0, 1]]))
        self.F_T_EEold = np.array(
            [[0.7071, 0.7071, 0.0, 0.0], [0.7071, -0.7071, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        self.NE_T_F = np.linalg.inv(
            np.array(
                [[0.7071, 0.7071, 0.0, 0.0], [-0.7071, 0.7071, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1034], [0.0, 0.0, 0.0, 1.0]]
            )
        )

    def inverse_kinematics(self, target_pos, target_orn, current_joint_state):
        R_T_EE = pos_orn_to_matrix(target_pos, target_orn)
        _target_pos, _target_orn = matrix_to_pos_orn(R_T_EE @ self.NE_T_F @ self.F_T_EEold)
        q_new = Kinematics.inverse(
            to_affine(_target_pos, _target_orn).vector(), current_joint_state
        )  # , self.null_space)
        q_new[np.where(q_new > 0)] %= 2 * np.pi
        q_new[np.where(q_new < 0)] %= -2 * np.pi
        if np.any(q_new < self.ll) or np.any(q_new > self.ul):
            print("Solution exceeds joint limits")
            return current_joint_state
        return q_new
