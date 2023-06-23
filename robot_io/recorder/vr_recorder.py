import logging
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

import numpy as np

from robot_io.utils.utils import depth_img_to_uint16

try:
    import pyttsx3

    from robot_io.utils.utils import TextToSpeech
except ModuleNotFoundError:

    class TextToSpeech:
        """
        Print text if TextToSpeech unavailable.
        """

        def say(self, x):
            print(x)


# A logger for this file
log = logging.getLogger(__name__)


def process_obs(obs):
    for key, value in obs.items():
        if "depth" in key:
            obs[key] = depth_img_to_uint16(obs[key])
    return obs


def count_previous_frames():
    return len(list(Path.cwd().glob("frame*.npz")))


class VrRecorder:
    """
    Save observations and robot trajectories when teleoperating the robot with an HTC Vive VR controller.
    Press button 1 to start and stop recording.
    Hold button 1 to delete the last recorded episode.

    Args:
        n_digits: Zero padding for files.
    """

    def __init__(self, n_digits):
        self.recording = False
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = count_previous_frames()
        self.tts = TextToSpeech()
        self.current_episode_filenames = []
        self.n_digits = n_digits
        self.delete_thread = None
        self.dead_man_switch_was_down = False
        try:
            # when adding to previous recording
            self.ep_start_end_ids = np.load("ep_start_end_ids.npy")
        except FileNotFoundError:
            self.ep_start_end_ids = []

    def step(self, action, obs, record_info):
        if record_info is None or "dead_man_switch_triggered" not in record_info:
            return
        if record_info["dead_man_switch_triggered"]:
            self.dead_man_switch_was_down = True
        if record_info["trigger_release"] and not self.recording and not self.is_deleting:
            if not self.dead_man_switch_was_down:
                self.tts.say("please press the dead man switch once before starting to record")
            else:
                self.recording = True
                self.tts.say("start recording")
                self.current_episode_filenames = []
                self.ep_start_end_ids.append([self.save_frame_cnt])
        elif record_info["trigger_release"] and self.recording:
            self.recording = False
            self.ep_start_end_ids[-1].append(self.save_frame_cnt)
            self.save(action, obs, True)
            self.tts.say("finish recording")
        if record_info["hold_event"]:
            if self.recording:
                self.recording = False
            self.delete_last_episode()
        if self.recording:
            assert action is not None
            self.save(action, obs, False)

    @property
    def is_deleting(self):
        return self.delete_thread is not None and self.delete_thread.is_alive()

    def delete_last_episode(self):
        self.delete_thread = threading.Thread(target=self._delete_last_episode, daemon=True)
        self.delete_thread.start()
        self.ep_start_end_ids = self.ep_start_end_ids[:-1]
        np.save("ep_start_end_ids.npy", self.ep_start_end_ids)

    def _delete_last_episode(self):
        log.info("Delete episode")
        while not self.queue.empty():
            log.info("Wait until files are saved")
            time.sleep(0.01)
        num_frames = len(self.current_episode_filenames)
        self.tts.say(f"Deleting last episode with {num_frames} frames")
        for filename in self.current_episode_filenames:
            os.remove(filename)
        self.tts.say("Finished deleting")
        self.save_frame_cnt -= num_frames
        self.current_episode_filenames = []

    def save(self, action, obs, done):
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.put((filename, action, obs, done))
        if done:
            np.save("ep_start_end_ids.npy", self.ep_start_end_ids)

    def process_queue(self):
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
            filename, action, obs, done = msg
            # change datatype of depth images to save storage space
            obs = process_obs(obs)
            np.savez_compressed(filename, **obs, action=action, done=done)

    def __enter__(self):
        """
            with ... as ... : logic
        Returns:
            None
        """
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """
            with ... as ... : logic
        Returns:
            None
        """
        if self.running:
            self.queue.put("QUIT")
            self.process.join()
