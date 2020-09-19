import enum

class Config:
    std_dev = 0.2
    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    
    
    total_time_steps = 1000000

    sample_rate = 0.1
    episode_sim_time = 50 # simulation time for a training episode
    steps_per_episode = int(episode_sim_time/sample_rate)

    number_of_episodes = int(total_time_steps/steps_per_episode)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005
    batch_size=64
    buffer_size=50000 
    show_env=True

    environment_desc = 'Pendulum-v0'

    exact = 'exact'
    euler = 'euler'
    method = exact