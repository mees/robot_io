import logging
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from robot_io.utils.utils import TextToSpeech

# A logger for this file
log = logging.getLogger(__name__)


class CalibRecorder:
    """
    Save robot poses for the calibration of a static camera.
    Press button 1 on HTC Vive VR controller to save a pose.
    """

    def __init__(self, n_digits, save_dir):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = 0
        self.tts = TextToSpeech()
        self.prev_done = False
        self.current_episode_filenames = []
        self.n_digits = n_digits
        self.save_dir = Path(save_dir)
        if len(list(self.save_dir.glob("*.npz"))):
            answer = input(f"{Path(os.getcwd()) / save_dir} not empty. Delete files? (Y/n)")
            if answer == "" or answer.lower() == "y":
                shutil.rmtree(self.save_dir)
            else:
                exit()
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, tcp_pose, marker_pose, record_info):
        """
        Save the pose information if button 1 is pressed.

        Args:
            tcp_pose: Robot pose as 4x4 matrix.
            marker_pose: Marker pose in static camera as 4x4 matrix.
            record_info: Dictionary with VR controller button states.
        """
        if record_info["trigger_release"]:
            self.tts.say("pose sampled")
            self._save(tcp_pose, marker_pose)

    def _save(self, tcp_pose, marker_pose):
        filename = self.save_dir / f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.put((filename, tcp_pose, marker_pose))

    def _process_queue(self):
        """
        Process function for queue.
        Returns:
            None
        """
        while True:
            msg = self.queue.get()
            if msg == "QUIT":
                self.running = False
                break
            filename, tcp_pose, marker_pose = msg
            print("saving ", filename, os.getcwd())
            np.savez(filename, tcp_pose=tcp_pose, marker_pose=marker_pose)

    def __enter__(self):
        """
            with ... as ... : logic
        Returns:
            None
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            with ... as ... : logic
        Returns:
            None
        """
        if self.running:
            self.queue.put("QUIT")
            self.process.join()
