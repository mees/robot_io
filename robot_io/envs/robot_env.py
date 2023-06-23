import time

import gym
import hydra
import numpy as np

from robot_io.input_devices.keyboard_input import keyboard_control
from robot_io.utils.utils import FpsController, restrict_workspace, timeit


class RobotEnv(gym.Env):
    """
    Example env class that can be used for teleoperation.
    Should be adapted to handle specific tasks.

    Args:
        robot: Robot interface.
        workspace_limits: Workspace limits defined as a bounding box or as hollow cylinder.
    """

    def __init__(
        self,
        robot,
        camera_manager_cfg,
        workspace_limits,
        freq: int = 15,
        show_fps: bool = False,
    ):
        self.robot = robot
        self.workspace_limits = workspace_limits
        self.camera_manager = hydra.utils.instantiate(camera_manager_cfg, robot_name=robot.name)
        self.show_fps = show_fps
        self.fps_controller = FpsController(freq)
        self.t1 = time.time()

    def reset(self, target_pos=None, target_orn=None, gripper_state="open"):
        """
        Reset robot to neutral position and reset gripper to initial gripper state.
        """
        self.robot.open_gripper(blocking=True)
        if target_pos is not None and target_orn is not None:
            success = self.robot.move_cart_pos_abs_ptp(target_pos, target_orn)
        else:
            success = self.robot.move_to_neutral()

        if not success:
            print("Robot cannot reach target pose. What do you want to do?")
            s = input("Press 'k' for resetting with the keyboard or 'n' to move to the neutral position.")
            if s == "n":
                self.robot.move_to_neutral()
            elif s == "k":
                keyboard_control(self)
            s = input("Do you want to retry moving to the target pose? Y/n")
            if s != "n":
                return self.reset(target_pos, target_orn, gripper_state)

        if gripper_state == "open":
            self.robot.open_gripper(blocking=True)
        elif gripper_state == "closed":
            self.robot.close_gripper(blocking=True)
        else:
            raise ValueError
        return self._get_obs()

    def _get_obs(self):
        """
        Get observation dictionary.
        Returns:
            Dictionary with image obs and state obs
        """
        obs = self.camera_manager.get_images()
        obs["robot_state"] = self.robot.get_state()
        return obs

    def get_reward(self, obs, action):
        return 0

    def get_termination(self, obs):
        return False

    def get_info(self, obs, action):
        info = {}
        return info

    def step(self, action):
        """
        Execute one action on the robot.

        Args:
            action (dict): {"motion": (position, orientation, gripper_action), "ref": "rel"/"abs"}
                a dict with the key 'motion' which is a cartesian motion tuple
                and the key 'ref' which specifies if the motion is absolute or relative
        Returns:
            obs (dict): agent's observation of the current environment.
            reward (float): Currently always 0.
            done (bool): whether the episode has ended, currently always False.
            info (dict): contains auxiliary diagnostic information, currently empty.
        """
        if action is None:
            return self._get_obs(), 0, False, {}
        assert isinstance(action, dict) and len(action["motion"]) == 3

        target_pos, target_orn, gripper_action = action["motion"]
        ref = action["ref"]

        if ref == "abs":
            target_pos = restrict_workspace(self.workspace_limits, target_pos)
            # TODO: use LIN for panda
            self.robot.move_async_cart_pos_abs_lin(target_pos, target_orn)
        elif ref == "rel":
            self.robot.move_async_cart_pos_rel_lin(target_pos, target_orn)
        else:
            raise ValueError

        if gripper_action == 1:
            self.robot.open_gripper()
        elif gripper_action == -1:
            self.robot.close_gripper()
        else:
            raise ValueError

        self.fps_controller.step()
        if self.show_fps:
            print(f"FPS: {1 / (time.time() - self.t1)}")
        self.t1 = time.time()

        obs = self._get_obs()

        reward = self.get_reward(obs, action)

        termination = self.get_termination(obs)

        info = self.get_info(obs, action)

        return obs, reward, termination, info

    def render(self, mode="human"):
        """
        Renders the environment.
        If mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.

        Args:
            mode (str): the mode to render with
        """
        if mode == "human":
            self.camera_manager.render()
            self.robot.visualize_joint_states()
            self.robot.visualize_external_forces()
