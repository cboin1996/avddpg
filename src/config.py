import enum

class Config:
    

    
    
    total_time_steps = 1000000

    sample_rate = 0.1
    episode_sim_time = 20 # simulation time for a training episode
    steps_per_episode = int(episode_sim_time/sample_rate)

    number_of_episodes = int(total_time_steps/steps_per_episode)
    
    gamma = 0.99 # Discount factor for future rewards

    
    # Learning rate for actor-critic models
    critic_lr = 0.001
    actor_lr = 0.0001
    std_dev = 0.02 # actor gaussian noise standard dev
    tau = 0.001 # target network update coeff

    batch_size=64
    buffer_size=500000
    show_env=False

    exact = 'exact'
    euler = 'euler'
    method = exact