import numpy as np

from robot_io.utils.utils import (
    angle_between,
    matrix_to_orn,
    orn_to_matrix,
    quat_to_euler,
    ReferenceType,
    restrict_workspace,
    to_tcp_frame,
    to_world_frame,
)


class RelActionControl:
    """
    This is a helper class that can be used in combination with any of the robot interfaces.
    It handles the conversion of relative actions to absolute actions.

    Args:
        workspace_limits: workspace limits defined as a bounding box or as hollow cylinder.
        relative_action_reference_frame: If "current", interpret relative action wrt. the current pose of the robot.
            This may lead to a drift if the load masses are not correctly configured.
            If "desired", interpret relative action as an update to a desired pose (setpoint).
        relative_action_control_frame: If "world", interpret relative action wrt. robot base frame (world frame).
            If "tcp" interpret relative action wrt. tcp frame.
        relative_pos_clip_threshold: When using relative_action_reference_frame="desired" clip the position part of
            the relative action if it deviates from the current robot position further than a threshold (x,y,z).
        relative_rot_clip_threshold: When using relative_action_reference_frame="desired" clip the orientation part of
            the relative action if it deviates from the current robot orientation further than a threshold  (α,β,γ).
        limit_control_5_dof: If True, only rotate the end-effector around its z-axis.
        max_ee_pitch: Limit the end-effector pitch (the angle between EE x-axis and world x-y-plane), in degree.
        max_ee_roll: Limit the end-effector roll (the angle between EE y-axis and world x-y-plane), in degree.
        ll: Lower joint limits in rad.
        ul: Upper joint limits in rad.
        default_orn_x: Default euler angle α around x-axis for 5-dof control.
        default_orn_y: Default euler angle β around y-axis for 5-dof control.
    """

    def __init__(
        self,
        workspace_limits,
        relative_action_reference_frame,
        relative_action_control_frame,
        relative_pos_clip_threshold,
        relative_rot_clip_threshold,
        limit_control_5_dof,
        max_ee_pitch,
        max_ee_roll,
        ll,
        ul,
        default_orn_x,
        default_orn_y,
    ):

        self.workspace_limits = workspace_limits
        assert relative_action_reference_frame in ["current", "desired"]
        self.relative_action_reference_frame = relative_action_reference_frame
        assert relative_action_control_frame in ["world", "tcp"]
        self.relative_action_control_frame = relative_action_control_frame
        self.relative_pos_clip_threshold = relative_pos_clip_threshold
        self.relative_rot_clip_threshold = relative_rot_clip_threshold
        self.limit_control_5_dof = limit_control_5_dof
        self.max_ee_pitch = np.radians(max_ee_pitch)
        self.max_ee_roll = np.radians(max_ee_roll)
        self.ll = np.array(ll)
        self.ul = np.array(ul)
        self.default_orn_x = default_orn_x
        self.default_orn_y = default_orn_y
        self.desired_pos, self.desired_orn = None, None

    def to_absolute(self, rel_action_pos, rel_action_orn, state, reference_type):
        """
        Convert relative action to absolute action.

        Args:
            rel_action_pos: Position increment (x,y,z) in meter.
            rel_action_orn: Orientation increment (α,β,γ) in rad.
            state: Robot state dict.
            reference_type (enum): RELATIVE, ABSOLUTE or JOINT.

        Returns:
            abs_target_pos: absolute target position (x,y,z).
            abs_target_orn: absolute target orientation (α,β,γ).
        """
        tcp_pos, tcp_orn, joint_positions = state["tcp_pos"], state["tcp_orn"], state["joint_positions"]
        if self.limit_control_5_dof:
            rel_action_orn = self._enforce_limit_gripper_joint(joint_positions, rel_action_orn)
            rel_action_orn[:2] = 0
        rel_action_pos, rel_action_orn = self._restrict_action_if_contact(rel_action_pos, rel_action_orn, state)

        if self.relative_action_reference_frame == "current":
            if self.relative_action_control_frame == "tcp":
                rel_action_pos, rel_action_orn = to_world_frame(rel_action_pos, rel_action_orn, tcp_orn)
            tcp_orn = quat_to_euler(tcp_orn)
            abs_target_pos = tcp_pos + rel_action_pos
            abs_target_orn = matrix_to_orn(orn_to_matrix(rel_action_orn) @ orn_to_matrix(tcp_orn))
            if self.limit_control_5_dof:
                abs_target_orn = self._enforce_5_dof_control(abs_target_orn)
            abs_target_pos = restrict_workspace(self.workspace_limits, abs_target_pos)
            return abs_target_pos, abs_target_orn
        else:
            if reference_type != ReferenceType.RELATIVE:
                self.desired_pos, self.desired_orn = tcp_pos, tcp_orn
                self.desired_orn = quat_to_euler(self.desired_orn)
                if self.limit_control_5_dof:
                    self.desired_orn = self._enforce_5_dof_control(self.desired_orn)
            if self.relative_action_control_frame == "tcp":
                rel_action_pos, rel_action_orn = to_world_frame(rel_action_pos, rel_action_orn, self.desired_orn)
            self.desired_pos, desired_orn = self._apply_to_desired_pose(
                rel_action_pos, rel_action_orn, tcp_pos, tcp_orn
            )
            self.desired_orn = self._restrict_orientation(desired_orn)
            self.desired_pos = restrict_workspace(self.workspace_limits, self.desired_pos)
            return self.desired_pos, self.desired_orn

    def _enforce_5_dof_control(self, abs_target_orn):
        """
        Maintain default end-effector orientation around x and y-axis for 5-dof control.

        Args:
            abs_target_orn: Absolute target orientation (α,β,γ).

        Returns:
            abs_target_orn: Absolute target orientation with default α and β component.
        """
        assert len(abs_target_orn) == 3
        abs_target_orn[0] = self.default_orn_x
        abs_target_orn[1] = self.default_orn_y
        return abs_target_orn

    def _enforce_limit_gripper_joint(self, joint_positions, rel_target_orn):
        """
        When using 5-dof control clip the rotation around the EE z-axis if it gets close to the joint limits of joint 7.

        Args:
            joint_positions: Current joint angles in rad.
            rel_target_orn: Relative target orientation (α,β,γ).

        Returns:
            Clipped relative target orientation.
        """
        if rel_target_orn[2] < 0 and joint_positions[-1] - rel_target_orn[2] > self.ul[-1] * 0.8:
            rel_target_orn[2] = 0
        elif rel_target_orn[2] > 0 and joint_positions[-1] - rel_target_orn[2] < self.ll[-1] * 0.8:
            rel_target_orn[2] = 0
        return rel_target_orn

    def _restrict_action_if_contact(self, rel_action_pos, rel_action_orn, state):
        """
        If the robot's contact force-torque thresholds are exceeded, set the relative action component which goes in the
        direction of contact to 0.

        Args:
            rel_action_pos: Position increment (x,y,z) in meter.
            rel_action_orn: Orientation increment (α,β,γ) in rad.
            state: Current robot state dict.

        Returns:
            rel_action_pos: Clipped position increment (x,y,z) in meter.
            rel_action_orn: Clipped orientation increment (α,β,γ) in rad.

        """
        if not np.any(state["contact"]):
            return rel_action_pos, rel_action_orn

        if self.relative_action_control_frame == "world":
            # rel_action_pos = np.linalg.inv(orn_to_matrix(state["tcp_orn"])) @ rel_action_pos
            rel_action_pos, rel_action_orn = to_tcp_frame(
                rel_action_pos, rel_action_orn, quat_to_euler(state["tcp_orn"])
            )
        for i in range(3):
            if state["contact"][i]:
                # check opposite signs
                if state["force_torque"][i] * rel_action_pos[i] < 0:
                    rel_action_pos[i] = 0
        for i in range(3):
            if state["contact"][i + 3]:
                # check opposite signs
                if state["force_torque"][i + 3] * rel_action_orn[i] < 0:
                    rel_action_orn[i] = 0
        if self.relative_action_control_frame == "world":
            rel_action_pos, rel_action_orn = to_world_frame(
                rel_action_pos, rel_action_orn, quat_to_euler(state["tcp_orn"])
            )
        return rel_action_pos, rel_action_orn

    def _restrict_orientation(self, abs_target_orn):
        """
        Restrict the end-effector pitch and roll i.e. the tilting of the end-effector.

        Args:
            abs_target_orn: Absolute target orientation (α,β,γ).

        Returns:
            abs_target_orn: Updated absolute target orientation (α,β,γ).
        """
        tcp_orn_mat = orn_to_matrix(abs_target_orn)
        tcp_x = tcp_orn_mat[:, 0]
        tcp_y = tcp_orn_mat[:, 1]
        # limit pitch (the angle between EE x-axis and world x-y-plane).
        if np.abs(np.radians(90) - angle_between(tcp_x, np.array([0, 0, 1]))) > self.max_ee_pitch:
            return self.desired_orn
        # limit roll (the angle between EE y-axis and world x-y-plane).
        if np.abs(np.radians(90) - angle_between(tcp_y, np.array([0, 0, 1]))) > self.max_ee_roll:
            return self.desired_orn
        return abs_target_orn

    def _apply_to_desired_pose(self, rel_action_pos, rel_action_orn, tcp_pos, tcp_orn):
        """
        Update the desired position and orientation with new relative action.

        Args:
            rel_action_pos: Position increment (x,y,z) in meter.
            rel_action_orn: Orientation increment (α,β,γ) in rad.
            tcp_pos: Current (measured) tcp position (x,y,z)
            tcp_orn: Current (measured) tcp orientation (α,β,γ).

        Returns:
            desired_pos: Updated desired absolute position.
            desired_orn: Updated desired absolute orientation.
        """
        # limit position
        desired_pos = self.desired_pos.copy()
        desired_orn = self.desired_orn.copy()
        for i in range(3):
            if rel_action_pos[i] > 0 and self.desired_pos[i] - tcp_pos[i] < self.relative_pos_clip_threshold:
                desired_pos[i] += rel_action_pos[i]
            elif rel_action_pos[i] < 0 and tcp_pos[i] - self.desired_pos[i] < self.relative_pos_clip_threshold:
                desired_pos[i] += rel_action_pos[i]
        # limit orientation
        rot_diff = quat_to_euler(matrix_to_orn(np.linalg.inv(orn_to_matrix(self.desired_orn)) @ orn_to_matrix(tcp_orn)))
        for i in range(3):
            if rel_action_orn[i] > 0 and rot_diff[i] >= self.relative_rot_clip_threshold:
                rel_action_orn[i] = 0
            elif rel_action_orn[i] < 0 and rot_diff[i] <= -self.relative_rot_clip_threshold:
                rel_action_orn[i] = 0
        desired_orn = matrix_to_orn(orn_to_matrix(rel_action_orn) @ orn_to_matrix(desired_orn))
        return desired_pos, desired_orn
