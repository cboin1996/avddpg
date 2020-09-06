class Config:
    std_dev = 0.2
    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    total_episodes = 100
    steps_per_episode = 50000
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005
    batch_size=64
    buffer_size=50000 
    show_env=True

    environment_desc = 'MountainCarContinuous-v0'
    # env = "Pendulum-v0"