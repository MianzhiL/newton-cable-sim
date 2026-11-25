def register():
    import gymnasium as gym
    from cable_robot_sample.teleop.compatibility import install

    install()

    name = 'Cable-Robot-Sample-Franka-v0'
    if name not in gym.registry:
        gym.register(id=name, entry_point='cable_robot_sample.teleop.recorded:RecordedCableEnv', disable_env_checker=True,
                     kwargs={'env_cfg_entry_point': 'cable_robot_sample.teleop.reset:make_config'})
    return []
