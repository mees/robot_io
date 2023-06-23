from ikfast_franka_panda import get_ik
import numpy as np

from robot_io.utils.utils import matrix_to_pos_orn, pos_orn_to_matrix, timeit


def to_list(pose):
    rot = pose[:3, :3].tolist()
    pos = pose[:3, 3].tolist()
    return pos, rot


class IKfast:
    """
    Use IKfast to get inverse kinematics solution.
    Currently, only available for franka panda, but generally works for any robot.
    IKfast in a fast analytic kinematics engine written in c++.
    Uses pybind11 for python bindings.
    Adapted from https://github.com/yijiangh/ikfast_pybind

    IKfast only works for robot arms with maximum 6 DOF, but redundant robots like the panda have 7 DOF.
    Thus, we call IKfast for different angles of the redundant DOF.
    In case of multiple solutions, choose best solution based on distance to reference configuration (either a rest pose
    or the current joint configuration).

    Args:
        rp: Rest pose used for null space handling.
        ll: Lower joint limits.
        ul: Upper joint limits.
        F_T_NE (np.ndarray): Transformation from flange to nominal end-effector, as set in the Franka Desk.
        weights: Weights for each DOF for picking the IK solution (higher weight of a DOF corresponds to less
            distance between solution and reference configuration for that DOF).
        num_angles: How many solutions to generate (with linearly spaced angles for the redundant DOF).
        use_rest_pose: If True, use rest pose to pick a solution, otherwise use current configuration.
    """

    def __init__(
        self,
        rp,
        ll,
        ul,
        F_T_NE,
        weights=(1, 1, 1, 1, 1, 1, 1),
        num_angles=50,
        use_rest_pose=True,
    ):
        self.ll = np.array(ll)
        self.ul = np.array(ul)
        self.rp = np.array(rp[:7])
        self.num_dof = len(self.ll)
        self.weights = weights
        self.num_angles = num_angles
        self.use_rest_pose = use_rest_pose
        # values from urdf that was used to generate ik fast solution
        F_T_NE_ikfast = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]])
        # this accounts for a new tcp frame (e.g. when using longer fingers)
        self.NE_T_NE_ikfast = np.linalg.inv(F_T_NE) @ F_T_NE_ikfast

    def inverse_kinematics(self, target_pos, target_orn, current_joint_state):
        """
        Get IKfast solution.

        Args:
            target_pos: Target position (x,y,z).
            target_orn: Target orientation as quaternion (x,y,z,w) or euler_angles (α,β,γ).
            current_joint_state: Current joint state (j1, ..., jn).

        Returns:
            Joint state of solution or current joint state if no solution was found.
        """
        sols = self._get_ik_solutions(target_pos, target_orn)
        feasible_sols = self._filter_solutions(sols)
        if len(feasible_sols) < 1:
            print("Did not find IK Solution")
            return current_joint_state
        if self.use_rest_pose:
            # choose solution according to weighted distance to rest pose
            return self._choose_best_solution(feasible_sols, self.rp, self.weights)
        else:
            # choose solution according to weighted distance to current joint positions
            return self._choose_best_solution(feasible_sols, current_joint_state, self.weights)

    def _get_ik_solutions(self, target_pos, target_orn):
        # transform from NE to NE_ikfast frame
        # weird hack, otherwise ik doesnt find solution
        target_pose = pos_orn_to_matrix(
            *matrix_to_pos_orn(pos_orn_to_matrix(target_pos, target_orn) @ self.NE_T_NE_ikfast)
        )
        # IK fast needs position and orientation as lists
        target_pos, target_orn = to_list(target_pose)
        sols = []
        # call ik fast for different angles of redundant DOF
        for q_6 in np.linspace(self.ll[-1], self.ul[-1], self.num_angles):
            sols += get_ik(target_pos, target_orn, [q_6])
        return sols

    def _check_solution(self, sol):
        # which joint positions of the solution are outside the joint limits
        out_ids = np.where(np.logical_or(sol < self.ll, sol > self.ul))[0]
        if len(out_ids) == 0:
            # valid solution
            return sol

        test_sol = np.array(sol)
        test_sol[out_ids] = 9999.0
        for i in out_ids:
            for add_ang in [-2.0 * np.pi, 2.0 * np.pi]:
                test_ang = sol[i] + add_ang
                if self.ul[i] >= test_ang >= self.ll[i]:
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.0):
            # valid solution
            return test_sol
        # not a valid solution
        return np.ones(self.num_dof) * 9999.0

    def _filter_solutions(self, sols):
        sols = np.array(sols)
        # filter IK solution with joint limits
        feasible_sols = np.apply_along_axis(self._check_solution, axis=1, arr=sols)
        return feasible_sols[~np.any(feasible_sols == 9999, axis=1)]

    @staticmethod
    def _choose_best_solution(sols, reference_q, weights):
        """
        Choose solution based on weighted difference to reference configuration.
        """
        best_sol_ind = np.argmin(np.sum((weights * (sols - np.array(reference_q))) ** 2, 1))
        return sols[best_sol_ind]
