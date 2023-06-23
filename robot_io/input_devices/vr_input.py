import logging
import sys
import time

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform.rotation import Rotation as R

# A logger for this file
from robot_io.utils.utils import pos_orn_to_matrix, restrict_workspace, z_angle_between

log = logging.getLogger(__name__)

GRIPPER_CLOSING_ACTION = -1
GRIPPER_OPENING_ACTION = 1

DEFAULT_RECORD_INFO = {
    "hold": False,
    "hold_event": False,
    "down": False,
    "dead_man_switch_triggered": False,
    "triggered": False,
    "trigger_release": False,
}


class VrInput:
    """
    Use HTC VIVE for tele-operation using Steam and Bullet VR.

    Args:
        workspace_limits: Workspace limits defined as a bounding box or as hollow cylinder.
        robot: Robot interface
        button_hold_threshold: After how many steps a button press counts as "hold".
    """

    def __init__(self, robot, workspace_limits, button_hold_threshold=60, tracking_error_threshold=0.03):
        self.robot = robot
        self.vr_controller_id = 3
        self.POSITION = 1
        self.ORIENTATION = 2
        self.ANALOG = 3
        self.BUTTONS = 6
        self.BUTTON_A = 2
        self.BUTTON_B = 1
        self.gripper_orientation_offset = R.from_euler("xyz", [0, 0, np.pi / 2])
        self.vr_pos_uid = None
        self.vr_coord_rotation = np.eye(4)
        self.workspace_limits = workspace_limits
        self.tracking_error_threshold = tracking_error_threshold
        self.has_tracking_error = False
        self.p = None
        self.button_1_press_counter = 0
        self.button_hold_threshold = button_hold_threshold
        self._initialize_bullet()

        self.prev_action = None
        self.robot_start_pos_offset = None
        self.out_of_workspace_offset = np.zeros(3)
        self.prev_button_info = DEFAULT_RECORD_INFO

        self.calibrate_vr_coord_system()

    def _initialize_bullet(self):
        """
        Connect to Bullet SHARED_MEMORY.
        """
        self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
        cid = self.p._client
        if cid < 0:
            log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
            sys.exit(1)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_PICKING, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        log.info(f"Connected to server with id: {cid}")

    def get_action(self):
        """
        Query the VR controller pose, transform to robot base frame and handle workspace limits.
        If no VR event arrived, return previous action.

        Returns:
            action (dict): EE target pos, orn and gripper action (in robot base frame).
            record_info (dict): Info with which buttons are pressed (for recording).
        """
        record_info = DEFAULT_RECORD_INFO
        vr_events = self.p.getVREvents()
        if vr_events != ():
            assert len(vr_events) == 1, "Only one VR controller should be turned on at the same time."
            for event in vr_events:
                # if event[0] == self.vr_controller_id:
                vr_action = self._vr_event_to_action(event)

                record_info = self._get_record_info(event)

                # if "dead man's switch" is not pressed, do not update pose
                if not self._dead_mans_switch_down(event):
                    return self.prev_action, record_info
                # reset button pressed
                elif self._dead_mans_switch_triggered(event):
                    self.has_tracking_error = False
                    self.prev_action = None
                    self._reset_vr_coord_system(vr_action)
                elif self.has_tracking_error:
                    return self.prev_action, record_info
                # transform pose from vr coord system to robot base frame
                robot_action = self._transform_action_vr_to_robot_base(vr_action)
                robot_action = {"motion": robot_action, "ref": "abs"}

                if self._check_tracking_error(robot_action):
                    self.has_tracking_error = True
                    log.error(f"Tracking error: {self.prev_action['motion'][0]}, {robot_action['motion'][0]}")
                    return self.prev_action, record_info

                self.prev_action = robot_action
        return self.prev_action, record_info

    def _check_tracking_error(self, current_action):
        """
        Check if there is a tracking error, i.e. the position difference of the current action and the previous action
        exceeds a threshold.

        Args:
            current_action (dict): Action at current time step.

        Returns:
            True if tracking error, False otherwise.
        """
        return self.prev_action is not None and np.any(
            np.abs(current_action["motion"][0] - self.prev_action["motion"][0]) > self.tracking_error_threshold
        )

    def _get_record_info(self, event):
        """
        Detect button events on VR controller.
        The detected events are:
        - down: If button 1 down.
        - triggered: If button 1 triggered.
        - trigger_release: If button 1 released.
        - hold: If `down` for more than `self.hold_threshold` steps.
        - hold_event: If `hold` and previous step was not `hold`.
        - dead_man_switch_triggered: If button 8 was triggered.

        Args:
            event: Bullet VR event.

        Returns:
            Record info dictionary.
        """
        button_1_hold = False
        if self._button_1_down(event):
            self.button_1_press_counter += 1
        else:
            self.button_1_press_counter = 0
        if self.button_1_press_counter >= self.button_hold_threshold:
            button_1_hold = True

        self.prev_button_info = {
            "hold_event": button_1_hold and not self.prev_button_info["hold"],
            "hold": button_1_hold,
            "down": self._button_1_down(event),
            "dead_man_switch_triggered": self._dead_mans_switch_triggered(event),
            "triggered": self._button_1_triggered(event) and self._button_1_down(event),
            "trigger_release": self._button_1_released(event)
            and not self.prev_button_info["hold"]
            and self.prev_button_info["down"],
        }
        return self.prev_button_info

    def _dead_mans_switch_down(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_IS_DOWN)

    def _dead_mans_switch_triggered(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_WAS_TRIGGERED)

    def _button_1_down(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_IS_DOWN)

    def _button_1_triggered(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_WAS_TRIGGERED)

    def _button_1_released(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_WAS_RELEASED)

    def _reset_vr_coord_system(self, vr_action):
        """
        This is called when the dead man's switch is triggered.
        The current vr controller pose is taken as origin for the proceeding vr motion.
        This is to ensure, that the robot does not jump to a new position or orientation if the dead man's switch
        is interrupted while the controller position changes.

        Args:
            vr_action (tuple): The current vr controller position, orientation and gripper action.
        """
        pos, orn = self.robot.get_tcp_pos_orn()
        T_VR = self.vr_coord_rotation @ pos_orn_to_matrix(vr_action[0], vr_action[1])

        self.robot_start_pos_offset = pos - T_VR[:3, 3]
        self.robot_start_orn_offset = R.from_matrix(T_VR[:3, :3]).inv() * R.from_quat(orn)

        self.out_of_workspace_offset = np.zeros(3)

    def _transform_action_vr_to_robot_base(self, vr_action):
        """
        Transform the vr controller pose to the coordinate system of the robot base.

        Args:
            vr_action (tuple): vr_pos, vr_orn, gripper action (in vr coord system).

        Returns:
            robot_pos: Position (x,y,z) in robot base frame.
            robot_orn: Orientation as quaternion (x,y,z,w) in robot base frame.
            gripper_action: Unchanged.
        """
        vr_pos, vr_orn, grip = vr_action
        # rotate vr coord system to align orientation with robot base frame
        T_VR_Controller = self.vr_coord_rotation @ pos_orn_to_matrix(vr_action[0], vr_action[1])
        # robot pos and orn are calculated relative to last reset
        robot_pos = T_VR_Controller[:3, 3] + self.robot_start_pos_offset
        robot_orn = R.from_matrix(T_VR_Controller[:3, :3]) * self.robot_start_orn_offset
        robot_orn = robot_orn.as_quat()

        robot_pos = self._enforce_workspace_limits(robot_pos)

        return robot_pos, robot_orn, grip

    def _enforce_workspace_limits(self, controller_pos):
        """
        Clip controller_pos to workspace limits. If out of workspace, set offset to keep controller position at the
        border of the workspace.

        Args:
            controller_pos:

        Returns:
            Clipped position.
        """
        controller_pos -= self.out_of_workspace_offset
        clipped_pos = restrict_workspace(self.workspace_limits, controller_pos)
        self.out_of_workspace_offset += controller_pos - clipped_pos
        return clipped_pos

    def _vr_event_to_action(self, event):
        """
        Parse VR event.

        Args:
            event: Pybullet VR event

        Returns:
            vr_controller_pos: Position (x,y,z) in VR frame.
            vr_controller_orn: Orientation as quaternion (x,y,z,w) in VR frame.
            gripper_action: Binary gripper action.
        """
        vr_controller_pos = np.array(event[self.POSITION])
        vr_controller_orn = np.array(event[self.ORIENTATION])
        controller_analogue_axis = event[self.ANALOG]

        gripper_action = GRIPPER_CLOSING_ACTION if controller_analogue_axis > 0.1 else GRIPPER_OPENING_ACTION
        return vr_controller_pos, vr_controller_orn, gripper_action

    def wait_for_start_button(self):
        """
        Wait until dead man's switch is pressed once.
        """
        log.info("wait for start button press")
        action = None
        while action is None:
            action = self.get_action()
            time.sleep(0.1)
        log.info("start button pressed")

    def calibrate_vr_coord_system(self):
        """
        Align the orientation of the VR coordinate system to the robot base frame by moving the VR controller
        in x-direction of the robot base frame.
        """
        start_pose = None
        end_pose = None
        log.info("------------------------------")
        log.info("Setting up VR coordinate axes.")
        log.info("Please do the following steps:")
        log.info("1. Take VR controller and stand at the position from where you want to tele-operate the robot.")
        log.info("2. Press dead man's switch once")
        log.info("3. Move the controller in positive X-direction of the robot base frame.")
        log.info("4. Press button 1 once.")
        log.info("------------------------------")
        while True:
            vr_events = self.p.getVREvents()
            if vr_events != ():
                for event in vr_events:
                    # if event[0] == self.vr_controller_id:
                    vr_action = self._vr_event_to_action(event)
                    if self._dead_mans_switch_down(event) and start_pose is None:
                        log.info("Now move vr controller in positive X-direction of the robot base frame.")
                        start_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])
                    elif self._button_1_down(event) and start_pose is not None and end_pose is None:
                        log.info("Finished setting up VR coordinate system.")
                        end_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])

                    if start_pose is not None and end_pose is not None:
                        self._set_vr_coord_transformation(start_pose, end_pose)
                        return

    def _set_vr_coord_transformation(self, start_pose, end_pose):
        """
        Calculate rotation between default VR coordinate system and user defined vr coordinate system.
        The x-axis of the user defined coordinate system is defined as the vector from start_pose to end_pose.

        Args:
            start_pose: VR controller pose at start of new x-axis (4x4 np.ndarray).
            end_pose: VR controller pose at end of new x-axis (4x4 np.ndarray).
        """
        new_x = end_pose[:2, 3] - start_pose[:2, 3]
        new_x = new_x / np.linalg.norm(new_x)
        old_x = np.array([1, 0])
        z_angle = z_angle_between(new_x, old_x)
        self.vr_coord_rotation = np.eye(4)
        self.vr_coord_rotation[:3, :3] = R.from_euler("z", [z_angle]).as_matrix()


def print_buttons(vr_input):
    print("sleep 3")
    time.sleep(3)
    print("enter loop")
    while True:
        action, info = vr_input.get_action()
        if info["triggered"]:
            print("triggered")
        if info["hold_event"]:
            print("hold_event")
        if info["trigger_release"]:
            print("trigger_release")
        time.sleep(0.01)


if __name__ == "__main__":
    vr = VrInput(robot=None)
    print_buttons(vr)
