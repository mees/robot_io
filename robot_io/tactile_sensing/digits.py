import threading
import time

import cv2
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import hydra


class Digits:
    """
    Interface class to get data from digits tactile camera.
    """

    def __init__(self, device_a: str, device_b: str, resolution: str, fps: str):
        print("Found digit sensors \n {}".format(DigitHandler.list_digits()))
        print("Supported streams: \n {}".format(DigitHandler.STREAMS))
        self.serial1 = device_a
        self.serial2 = device_b
        self.resolution = resolution
        self.fps = fps
        self.d1 = Digit(self.serial1)
        self.d2 = Digit(self.serial2)
        self.d1.connect()
        self.d2.connect()
        print("Connected digits")
        assert resolution == "VGA" or resolution == "QVGA"
        self.d1.set_resolution(DigitHandler.STREAMS[resolution])
        self.d2.set_resolution(DigitHandler.STREAMS[resolution])
        assert (fps == "60fps" or fps == "30fps") or fps == "15fps"
        self.d1.set_fps(DigitHandler.STREAMS[resolution]["fps"][fps])
        self.d2.set_fps(DigitHandler.STREAMS[resolution]["fps"][fps])
        print("Finished initializing digits")
        print(self.d1.info())

    def get_image(self):
        frame1 = self.d1.get_frame()
        frame2 = self.d2.get_frame()
        return frame1, frame2

    def get_serials(self):
        return self.serial1, self.serial2


class _DigitsThread(threading.Thread):
    def __init__(self, digits_cfg):
        threading.Thread.__init__(self)
        self.digit = hydra.utils.instantiate(digits_cfg)
        self.flag_exit = False
        self.frame1 = None
        self.frame2 = None

    def run(self):
        while not self.flag_exit:
            self.frame1, self.frame2 = self.digit.get_image()


class ThreadedDigits:
    def __init__(self, digits_cfg):
        self._digit_thread = _DigitsThread(digits_cfg)
        self._digit_thread.start()

    def get_image(self):
        while self._digit_thread.frame1 is None or self._digit_thread.frame2 is None:
            time.sleep(0.01)
        frame1 = self._digit_thread.frame1.copy()
        frame2 = self._digit_thread.frame2.copy()
        return frame1, frame2

    def get_serials(self):
        return self._digit_thread.digit.get_serials()

    def __del__(self):
        self._digit_thread.flag_exit = True
        self._digit_thread.join()


def test_digit():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("../conf/tactile_sensing/digits.yaml")
    d = ThreadedDigits(cfg)
    while True:
        frame1, frame2 = d.get_image()
        cv2.imshow(f"Digit View {d.get_serials()[0]}", frame1)
        cv2.imshow(f"Digit View {d.get_serials()[1]}", frame2)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_digit()
