import logging
from multiprocessing import Event, Value
import re
import socket
import struct
import threading
import time

import numpy as np

command_dict = {
    "20": "Homing",
    "21": "Move Fingers",
    "22": "Stop",
    "24": "ACK Fast Stop",
    "25": "Grasp",
    "26": "Release Part",
    "43": "Get Opening Width",
    "45": "Get Force",
}

error_codes = [
    "E_SUCCESS",
    "E_NOT_AVAILABLE",
    "E_NO_SENSOR",
    "E_NOT_INITIALIZED",
    "E_ALREADY_RUNNING",
    "E_FEATURE_NOT_SUPPORTED",
    "E_INCONSISTENT_DATA",
    "E_TIMEOUT",
    "E_READ_ERROR",
    "E_WRITE_ERROR",
    "E_INSUFFICIENT_RESOURCES",
    "E_CHECKSUM_ERROR",
    "E_NO_PARAM_EXPECTED",
    "E_NOT_ENOUGH_PARAMS",
    "E_CMD_UNKNOWN",
    "E_CMD_FORMAT_ERROR",
    "E_ACCESS_DENIED",
    "E_ALREADY_OPEN",
    "E_CMD_FAILED",
    "E_CMD_ABORTED",
    "E_INVALID_HANDLE",
    "E_NOT_FOUND",
    "E_NOT_OPEN",
    "E_IO_ERROR",
    "E_INVALID_PARAMETER",
    "E_INDEX_OUT_OF_BOUNDS",
    "E_CMD_PENDING",
    "E_OVERRUN",
    "E_RANGE_ERROR",
    "E_AXIS_BLOCKED",
    "E_FILE_EXISTS",
]


class WSG50Controller:
    def __init__(self, max_opening_width=100, min_opening_width=0):
        self._request_opening_width_and_force = Event()
        self._open_gripper_event = Event()
        self._close_gripper_event = Event()
        self._stop_event = Event()
        self._opening_width = Value("d", -1)
        self._force = Value("d", -1)
        self._controller_thread = GripperControllerThread(
            max_opening_width,
            min_opening_width,
            self._request_opening_width_and_force,
            self._close_gripper_event,
            self._open_gripper_event,
            self._stop_event,
            self._opening_width,
            self._force,
        )
        self._controller_thread.start()
        # compatibility with simulated WSG50 class
        self.OPEN_ACTION = 1
        self.CLOSE_ACTION = -1

    def request_opening_width_and_force(self):
        self._request_opening_width_and_force.set()

    def open_gripper(self):
        self._close_gripper_event.clear()
        self._stop_event.set()
        self._open_gripper_event.set()

    def close_gripper(self):
        self._open_gripper_event.clear()
        self._stop_event.set()
        self._close_gripper_event.set()

    def get_opening_width(self):
        return self._opening_width.value

    def get_force(self):
        return self._force.value

    def home(self):
        self._controller_thread.home()


class GripperControllerThread(threading.Thread):
    def __init__(
        self,
        max_opening_width,
        min_opening_width,
        request,
        close_gripper_event,
        open_gripper_event,
        stop_event,
        opening_width,
        force,
    ):
        # self.opening_width_offset = 0.004762348175048828 # without rubber
        self.opening_width_offset = 0.0088
        self.max_opening_width = max_opening_width
        self.min_opening_width = min_opening_width
        self.own_address = ("localhost", 50601)
        self.gripper_address = "192.168.42.20"
        self.gripper_port = 6666
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.socket.connect((self.gripper_address, self.gripper_port))
        self.ack_fast_stop()
        self.opening_width = opening_width
        self.force = force
        self.opening_width.value = self._get_opening_width()
        self.force.value = self._get_force()
        self.request_opening_width_and_force = request
        self.close_gripper_event = close_gripper_event
        self.open_gripper_event = open_gripper_event
        self.stop_event = stop_event
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        while 1:
            if self.request_opening_width_and_force.is_set():
                self._send_get_opening_width()
                self._send_get_force()
                opening_width, force = self.get_opening_width_and_force()
                self.opening_width.value, self.force.value = opening_width, force
                self.request_opening_width_and_force.clear()
            if self.stop_event.is_set():
                self.stop()
                self.stop_event.clear()
            else:
                if self.close_gripper_event.is_set():
                    self.close_gripper()
                    self.close_gripper_event.clear()
                elif self.open_gripper_event.is_set():
                    self.open_gripper()
                    self.open_gripper_event.clear()
            time.sleep(0.001)

    def _split_messages(self, msg):
        msgs = re.split("aaaaaa", msg.hex())
        msgs = list(filter(None, msgs))
        result = []
        for m in msgs:
            result.append(self._parse_gripper_msg(bytes.fromhex(m)))
        return result

    def _parse_gripper_msg(self, msg):
        command_id = msg[0:1].hex()
        if command_id in command_dict:
            command_id = command_dict[command_id]
        size = struct.unpack("<H", msg[1:3])[0]  # in bytes
        error_code = error_codes[struct.unpack("<H", msg[3:5])[0]]
        params = None
        if size > 2:
            params = msg[5 : 3 + size]
            if command_id == "Get Opening Width":
                params = struct.unpack("f", params)[0]
            elif command_id == "Get Force":
                params = struct.unpack("f", params)[0]
        # print(command_id, size, error_code, params)
        return command_id, size, error_code, params

    def _send_msg(self, hex_msg):
        msg = bytearray.fromhex(hex_msg)
        self.socket.send(bytes(msg))

    def recv_msgs(self):
        return self._split_messages(self.socket.recvfrom(256)[0])

    def ack_fast_stop(self):
        preamble = "AAAAAA"
        command_id = "24"
        size = "0300"
        payload = "61636B"
        checksum = "0000"
        msg = preamble + command_id + size + payload + checksum
        self._send_msg(msg)
        self.recv_msgs()

    def open_gripper(self):
        logging.info("gripper controller: open")
        self.release_part(self.max_opening_width, 420)

    def close_gripper(self):
        logging.info("gripper controller: close")
        self.move_fingers(self.min_opening_width, 420)
        # self.grasp_part(0, 50)

    def home(self):
        preamble = "AAAAAA"
        command_id = "20"
        size = "0100"
        payload = "00"
        checksum = "0000"
        home_msg = preamble + command_id + size + payload + checksum
        self._send_msg(home_msg)
        # self.recv_msgs()

    def stop(self):
        logging.info("gripper controller: stop")
        preamble = "AAAAAA"
        command_id = "22"
        size = "0000"
        payload = ""
        checksum = "0000"
        stop_msg = preamble + command_id + size + payload + checksum
        self._send_msg(stop_msg)
        while True:
            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Stop" and answer[2] == "E_SUCCESS":
                    return

    def grasp_part(self, width=30.0, speed=200.0):
        # self.stop()
        preamble = "AAAAAA"
        command_id = "25"
        size = "0800"
        payload = np.array([width, speed], dtype=np.float32).tobytes().hex()
        checksum = "0000"
        grasp_msg = preamble + command_id + size + payload + checksum
        self._send_msg(grasp_msg)
        while True:

            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Grasp" and (answer[2] == "E_CMD_PENDING" or answer[2] == "E_ALREADY_RUNNING"):
                    print(answer)
                    return
                elif answer[0] == "Grasp" and answer[2] == "E_CMD_ABORTED":
                    self._send_msg(grasp_msg)
                else:
                    print(answer)

    def release_part(self, width=75.0, speed=200.0):
        logging.info("gripper controller: release_part")
        # self.stop()
        preamble = "AAAAAA"
        command_id = "26"
        size = "0800"
        payload = np.array([width, speed], dtype=np.float32).tobytes().hex()
        checksum = "0000"
        msg = preamble + command_id + size + payload + checksum
        self._send_msg(msg)
        while True:

            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Release Part" and (answer[2] == "E_CMD_PENDING" or answer[2] == "E_ALREADY_RUNNING"):
                    return
                elif answer[0] == "Release Part" and answer[2] == "E_CMD_ABORTED":
                    self._send_msg(msg)
                else:
                    print(answer)

    def move_fingers(self, width=75.0, speed=200.0):
        # width > 50 means open otherwise probably close
        logging.info("gripper controller: move_fingers {}".format(width))
        # self.stop()
        preamble = "AAAAAA"
        command_id = "21"
        size = "0900"
        payload = "00"
        payload += np.array([width, speed], dtype=np.float32).tobytes().hex()
        checksum = "0000"
        msg = preamble + command_id + size + payload + checksum
        self._send_msg(msg)
        while True:

            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Move Fingers" and (answer[2] == "E_CMD_PENDING" or answer[2] == "E_ALREADY_RUNNING"):
                    return
                elif answer[0] == "Move Fingers" and answer[2] == "E_CMD_ABORTED":
                    self._send_msg(msg)
                else:
                    print(answer)
        # print()

    def _get_opening_width(self):
        preamble = "AAAAAA"
        command_id = "43"
        size = "0300"
        payload = "000000"
        checksum = "0000"
        opening_msg = preamble + command_id + size + payload + checksum
        self._send_msg(opening_msg)
        while 1:
            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Get Opening Width":
                    # scale to mm and consider
                    opening_width = answer[3] * 0.001 - self.opening_width_offset
                    return np.clip(opening_width, 0, 1)

    def _get_force(self):
        preamble = "AAAAAA"
        command_id = "45"
        size = "0300"
        payload = "000000"
        checksum = "0000"
        msg = preamble + command_id + size + payload + checksum
        self._send_msg(msg)
        while 1:
            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Get Force":
                    return answer[3]

    def _send_get_opening_width(self):
        preamble = "AAAAAA"
        command_id = "43"
        size = "0300"
        payload = "000000"
        checksum = "0000"
        opening_msg = preamble + command_id + size + payload + checksum
        self._send_msg(opening_msg)

    def _send_get_force(self):
        preamble = "AAAAAA"
        command_id = "45"
        size = "0300"
        payload = "000000"
        checksum = "0000"
        msg = preamble + command_id + size + payload + checksum
        self._send_msg(msg)

    def get_opening_width_and_force(self):
        opening_width = None
        force = None
        while opening_width is None or force is None:
            answers = self.recv_msgs()
            for answer in answers:
                if answer[0] == "Get Force":
                    force = answer[3]
                elif answer[0] == "Get Opening Width":
                    # scale to mm and consider
                    opening_width = np.clip(answer[3] * 0.001 - self.opening_width_offset, 0, 1)
        return opening_width, force


if __name__ == "__main__":
    gripper = WSG50Controller(max_opening_width=77)
    gripper.open_gripper()
    time.sleep(2)
    gripper.request_opening_width_and_force()
    time.sleep(2)
    print("opening_width: {}".format(gripper.get_opening_width()))

    # gripper.open_gripper()
    # for j in range(50):
    #     gripper.request_opening_width_and_force()
    #     i = np.random.randint(0, 2)
    #     if i % 2 == 0:
    #         print("open")
    #         gripper.open_gripper()
    #     else:
    #         print("close")
    #         gripper.close_gripper()
    #     time.sleep(0.05)
    #     print("opening_width: {}".format(gripper.get_opening_width()))
    #     print("force: {}".format(gripper.get_force()))
    #     print()
    # time.sleep(3)
