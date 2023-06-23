from pathlib import Path

import hydra
import numpy as np

from robot_io.utils.utils import to_relative, to_relative_action_dict

N_DIGITS = 6
MAX_REL_POS = 0.02
MAX_REL_ORN = 0.05


def get_ep_start_end_ids(path):
    return np.sort(np.load(Path(path) / "ep_start_end_ids.npy"), axis=0)


def get_frame(path, i):
    filename = Path(path) / f"episode_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def reset(env, path, i):
    data = get_frame(path, i)
    robot_state = data["robot_obs"]
    gripper_state = "open" if robot_state[6] > 0.07 else "closed"
    env.reset(
        target_pos=robot_state[:3],
        target_orn=robot_state[3:6],
        gripper_state=gripper_state,
    )


def get_action(path, i, use_rel_actions=False, control_frame="tcp"):
    frame = get_frame(path, i)
    if control_frame == "tcp":
        action = frame["rel_actions_gripper"]
    else:
        action = frame["rel_actions_world"]
    print(action)
    pos = action[:3] * MAX_REL_POS
    orn = action[3:6] * MAX_REL_ORN

    new_action = {"motion": (pos, orn, action[-1]), "ref": "rel"}
    return new_action
    # if use_rel_actions:
    #     prev_action = get_frame(path, i - 1)['rel_action'].item()
    #     return to_relative_action_dict(prev_action, action)
    # else:
    #     return action


def to_relative_action_dict(pos, next_pos, gripper_action):
    pos = pos[:3]
    orn = next_pos[3:6]

    next_pos = next_pos[:3] * MAX_REL_POS
    next_orn = next_pos[3:6] * MAX_REL_ORN
    rel_pos, rel_orn = to_relative(pos, orn, next_pos, next_orn)
    action = {"motion": (rel_pos, rel_orn, gripper_action), "ref": "rel"}
    return action


def get_action_pos(path, i, use_rel_actions=False, control_frame="tcp"):
    frame = get_frame(path, i)
    robot_obs = frame["robot_obs"].item()
    if use_rel_actions:
        next_obs = get_frame(path, i + 1)["robot_obs"].item()
        gripper_action = robot_obs[-1]
        return to_relative_action_dict(robot_obs, next_obs, gripper_action)
    else:
        if control_frame == "tcp":
            action = frame["rel_actions_gripper"]
        else:
            action = frame["rel_actions_world"]
        pos = action[:3] * MAX_REL_POS
        orn = action[3:6] * MAX_REL_ORN
        new_action = {"motion": (pos, orn, action[-1]), "ref": "rel"}
        return new_action


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
    for start_idx, end_idx in ep_start_end_ids:
        reset(env, cfg.load_dir, start_idx)
        for i in range(start_idx + 1, end_idx + 1):
            action = get_action(
                cfg.load_dir, i, use_rel_actions, cfg.robot.rel_action_params.relative_action_control_frame
            )
            obs, _, _, _ = env.step(action)
            env.render()
        break


if __name__ == "__main__":
    main()
