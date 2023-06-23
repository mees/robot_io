from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np

from robot_io.utils.utils import to_relative_action_dict, to_relative_action_pos_dict

N_DIGITS = 6


def get_ep_start_end_ids(path):
    return np.sort(np.load(Path(path) / "ep_start_end_ids.npy"), axis=0)


def get_frame(path, i):
    filename = Path(path) / f"frame_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def reset(env, path, i):
    data = get_frame(path, i)
    robot_state = data["robot_state"].item()
    gripper_state = "open" if robot_state["gripper_opening_width"] > 0.07 else "closed"
    env.reset(
        target_pos=robot_state["tcp_pos"],
        target_orn=robot_state["tcp_orn"],
        gripper_state=gripper_state,
    )


def get_action(path, i, use_rel_actions=False):
    frame = get_frame(path, i)
    action = frame["action"].item()
    if use_rel_actions:
        prev_action = get_frame(path, i - 1)["action"].item()
        return to_relative_action_dict(prev_action, action)
    else:
        return action


def get_action_pos(path, i, use_rel_actions=False):
    frame = get_frame(path, i)
    pos = frame["robot_state"].item()
    action = frame["action"].item()

    if use_rel_actions:
        next_pos = get_frame(path, i + 1)["robot_state"].item()
        gripper_action = action["motion"][-1]
        return to_relative_action_pos_dict(pos, next_pos, gripper_action)
    else:
        return action


def get_action_pos_action(path, i, use_rel_actions=True):
    frame = get_frame(path, i)
    pose = frame["robot_state"].item()
    action = frame["action"].item()

    if use_rel_actions:
        next_pose = {"tcp_pos": action["motion"][0], "tcp_orn": action["motion"][1]}
        gripper_action = action["motion"][-1]
        return to_relative_action_pos_dict(pose, next_pose, gripper_action)
    else:
        return frame


@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):
    """
    Replay a recorded trajectory, either with absolute actions or relative actions.

    Args:
        cfg: Hydra config
    """
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    use_rel_actions = cfg.use_rel_actions
    ep_start_end_ids = get_ep_start_end_ids(cfg.load_dir)

    env.reset()
    get_relative_actions = {"pp": get_action_pos, "aa": get_action, "pa": get_action_pos_action}

    for start_idx, end_idx in ep_start_end_ids:
        reset(env, cfg.load_dir, start_idx)
        frame = get_frame(cfg.load_dir, 0)
        # plt.imshow(frame["rgb_static"])
        # plt.show()
        for i in range(start_idx + 1, end_idx + 1):
            action = get_relative_actions[cfg.rel_action_type](cfg.load_dir, i, use_rel_actions)
            obs, _, _, _ = env.step(action)
            env.render()


if __name__ == "__main__":
    main()
