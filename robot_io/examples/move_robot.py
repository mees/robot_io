import hydra
import numpy as np


@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):
    """
    Starting from the neutral position, move the EE left and right.

    Args:
        cfg: Hydra config.
    """
    robot = hydra.utils.instantiate(cfg.robot)
    robot.move_to_neutral()

    pos, orn = robot.get_tcp_pos_orn()

    left_pos = pos + np.array([0, -0.1, 0])
    right_pos = pos + np.array([0, 0.1, 0])

    while True:
        for p in (left_pos, right_pos):
            if not robot.move_cart_pos_abs_ptp(p, orn):
                exit()


if __name__ == "__main__":
    main()
