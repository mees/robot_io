import os
from pathlib import Path

import hydra
import numpy as np
from tqdm import tqdm

from robot_io.utils.utils import depth_img_from_uint16, quat_to_euler, to_relative_all_frames

TRAINING_DIR: str = "training"
VAL_DIR: str = "validation"
N_DIGITS = 6
FILENAME = "episode"
MAX_REL_POS = 0.02
MAX_REL_ORN = 0.05


def listdirs(rootdir):
    list_dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if (path / "ep_start_end_ids.npy").is_file():
                list_dirs.append(path)
            else:
                result = listdirs(path)
                if result:
                    list_dirs.append(result)
    if list_dirs:
        return list_dirs


def get_frame(path, i):
    filename = Path(path) / f"frame_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def get_filename(path, i):
    return Path(path) / f"{FILENAME}_{i:0{N_DIGITS}d}.npz"


def load_data(data):
    robot_state = data["robot_state"][()]
    action = data["action"][()]["motion"]
    tcp_pos = robot_state["tcp_pos"]
    tcp_orn = quat_to_euler(robot_state["tcp_orn"])
    gripper_width = robot_state["gripper_opening_width"]
    joint_positions = robot_state["joint_positions"]
    gripper_action = action[-1]  # int(gripper_width >= 0.078)
    # todo: include force torque
    robot_obs = np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])
    return tcp_pos, tcp_orn, action, robot_obs


def load_imgs(data, cam_num):
    suffix = [""]
    if cam_num > 1:
        suffix = [f"_{i}" for i in range(cam_num)]
    rgb_gripper = data["rgb_gripper"] if "rgb_gripper" in data else None
    depth_gripper = data["depth_gripper"] if "depth_gripper" in data else None
    rgb_static_list = [data["rgb_static" + suf] for suf in suffix]
    depth_static_list = [data["depth_static" + suf] for suf in suffix]
    return rgb_static_list, depth_static_list, rgb_gripper, depth_gripper


def compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, gripper_action):
    rel_pos_orn_dct = to_relative_all_frames(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn)
    rel_pos_orn_dct_new = {}
    for frame, (rel_pos, rel_orn) in rel_pos_orn_dct.items():
        # clipped_rel_pos = np.clip(rel_pos, -MAX_REL_POS, MAX_REL_POS) / MAX_REL_POS
        # clipped_rel_orn = np.clip(rel_orn, -MAX_REL_ORN, MAX_REL_ORN) / MAX_REL_ORN
        clipped_rel_pos, clipped_rel_orn = rel_pos / MAX_REL_POS, rel_orn / MAX_REL_ORN
        rel_action = np.concatenate([clipped_rel_pos, clipped_rel_orn, [gripper_action]])
        rel_pos_orn_dct_new[frame] = rel_action
    return rel_pos_orn_dct_new


def process_data(recording_dir, i, cam_num):
    data_prev = get_frame(recording_dir, i - 1)
    data_cur = get_frame(recording_dir, i)
    tcp_pos, tcp_orn, curr_action, robot_obs = load_data(data_cur)
    past_tcp_pos, past_tcp_orn, past_action, _ = load_data(data_prev)
    curr_gripper_action = curr_action[-1]
    tcp_pos_action = curr_action[0]
    tcp_orn_action = curr_action[1]
    past_tcp_pos_action = past_action[0]
    past_tcp_orn_action = past_action[1]
    rel_action = compute_rel_action(
        past_tcp_pos_action,
        past_tcp_orn_action,
        tcp_pos_action,
        tcp_orn_action,
        curr_gripper_action,
    )
    # rel_action = compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, curr_gripper_action)
    rgb_static_list, depth_static_list, rgb_gripper, depth_gripper = load_imgs(data_cur, cam_num)
    depth_static_list = [depth_img_from_uint16(depth_static, max_depth=4) for depth_static in depth_static_list]
    depth_gripper = depth_img_from_uint16(depth_gripper, max_depth=4) if depth_gripper is not None else None
    if np.linalg.norm(rel_action["world_frame"][3:6]) > 1.5 or np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:
        if np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:
            print("Displacement - out of bounds", i, rel_action["world_frame"])
        else:
            print("Rotation - out of bounds", i, rel_action["world_frame"])
        data_next = get_frame(recording_dir, i + 1)
        next_tcp_pos, next_tcp_orn, _, _ = load_data(data_next)
        rel_action = compute_rel_action(tcp_pos, tcp_orn, next_tcp_pos, next_tcp_orn, curr_gripper_action)
        if np.linalg.norm(rel_action["world_frame"][3:6]) > 1.5 or np.linalg.norm(rel_action["world_frame"][:3]) > 1.25:
            print("Still out of bounds", i, rel_action["world_frame"])
        else:
            print("Solved, in bounds", i, rel_action["world_frame"])
    curr_action = np.concatenate([curr_action[0], quat_to_euler(curr_action[1]), [curr_action[2]]])
    if cam_num > 1:
        depth_static_dict = {f"depth_static_{i}": depth_static for i, depth_static in enumerate(depth_static_list)}
        rgb_static_dict = {f"rgb_static_{i}": rgb_static for i, rgb_static in enumerate(rgb_static_list)}
        save_data = {
            "actions": curr_action,
            "rel_actions_world": rel_action["world_frame"],
            "rel_actions_gripper": rel_action["gripper_frame"],
            "robot_obs": robot_obs,
            "rgb_gripper": rgb_gripper,
            "depth_gripper": depth_gripper,
            **depth_static_dict,
            **rgb_static_dict,
        }
    else:
        save_data = {
            "actions": curr_action,
            "rel_actions_world": rel_action["world_frame"],
            "rel_actions_gripper": rel_action["gripper_frame"],
            "robot_obs": robot_obs,
            "rgb_static": rgb_static_list[0],
            "depth_static": depth_static_list[0],
            "rgb_gripper": rgb_gripper,
            "depth_gripper": depth_gripper,
        }
    return save_data


def create_dataset(recording_dirs, output_dir, cam_num):
    ep_start_end_ids_all = []
    min_max_action_values = {
        "rel_actions_world": [np.ones(7) * np.inf, -np.ones(7) * np.inf],
        "rel_actions_gripper": [np.ones(7) * np.inf, -np.ones(7) * np.inf],
    }
    new_end_idx = 0
    for recording_dir in tqdm(recording_dirs):
        ep_start_end_ids = np.sort(np.load(recording_dir / "ep_start_end_ids.npy"))
        for start_idx, end_idx in tqdm(ep_start_end_ids, leave=False):
            new_start_idx = new_end_idx
            for i in range(start_idx + 1, end_idx + 1):
                save_data = process_data(recording_dir, i, cam_num)
                for key in min_max_action_values.keys():
                    key_min, key_max = min_max_action_values[key]
                    min_max_action_values[key][0] = np.minimum(save_data[key], key_min)
                    min_max_action_values[key][1] = np.maximum(save_data[key], key_max)
                np.savez_compressed(get_filename(output_dir, new_end_idx), **save_data)
                new_end_idx += 1
            ep_start_end_ids_all.append((new_start_idx, new_end_idx - 1))
        for k, v in min_max_action_values.items():
            print(k, v)
    np.save(output_dir / "ep_start_end_ids.npy", ep_start_end_ids_all)


@hydra.main(config_path="../conf", config_name="preprocess_data")
def main(cfg):
    dataset_root = cfg.dataset_root
    output_dir = cfg.output_dir
    cam_num = cfg.cam_num
    list_recording_dirs = listdirs(dataset_root)
    # now flatten list of lists
    print(list_recording_dirs)
    recording_dirs = [item for sublist in list_recording_dirs for item in sublist]
    print("Found following subfolders containing recordings: ", recording_dirs)
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"The output dir {str(output_dir)} already exists. Do you want to overwrite? (Y/N)")
        c = input()
        if c == "Y" or c == "y":
            os.rmdir(output_dir)
        elif c == "N" or c == "n":
            return
    os.makedirs(output_dir, exist_ok=True)  # TODO: set false
    create_dataset(recording_dirs, output_dir, cam_num)


if __name__ == "__main__":
    main()
