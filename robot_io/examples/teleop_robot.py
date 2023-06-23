import hydra


@hydra.main(config_path="../conf")
def main(cfg):
    """
    Teleoperate the robot with different input devices.
    Depending on the recorder, either record the whole interaction or only if the recording is triggered by the input
    device.

    Args:
        cfg: Hydra config
    """
    recorder = hydra.utils.instantiate(cfg.recorder)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    obs = env.reset()
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)

    while True:
        action, record_info = input_device.get_action()
        next_obs, _, _, _ = env.step(action)
        recorder.step(action, obs, record_info)
        env.render()
        obs = next_obs


if __name__ == "__main__":
    main()
