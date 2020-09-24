import enum
import os, sys
class Config:
    def __init__(self):
        self.total_time_steps = 500000

        self.sample_rate = 0.1
        self.episode_sim_time = 20 # simulation time for a training episode
        self.steps_per_episode = int(self.episode_sim_time/self.sample_rate)

        self.number_of_episodes = int(self.total_time_steps/self.steps_per_episode)
        
        self.gamma = 0.99 # Discount factor for future rewards

        
        # Learning rate for actor-critic models
        self.critic_lr = 0.001
        self.actor_lr = 0.0001
        self.std_dev = 0.02 # actor gaussian noise standard dev
        self.tau = 0.001 # target network update coeff

        self.batch_size=64
        self.buffer_size=100000
        self.show_env=False

        self.exact = 'exact'
        self.euler = 'euler'
        self.method = self.exact

        self.res_dir = 'res'
        self.best_dir = 'best'

        self.actor_dir = 'actor.h5'
        self.actor_weights = 'actor_weights.h5'
        self.critic_dir = 'critic.h5'
        self.critic_weights = 'critic_weights.h5'
        self.t_actor_dir = 'target_actor.h5'
        self.t_actor_weights = 'target_actor_weights.h5'
        self.t_critic_dir = 'target_critic.h5'
        self.t_critic_weights = 'target_critic_weights.h5'

        self.fig_path = "res.png"
        self.param_path = "conf.txt"
        self.env_path = "env.txt"

        self.best_actor_conf = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.actor_dir)
        self.best_actor_weights = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.actor_weights)

        self.best_critic_conf = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.critic_dir)
        self.best_critic_weights = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.critic_weights)

        self.best_tactor_conf = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.t_actor_dir)
        self.best_tactor_weights = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.t_actor_weights)

        self.best_tcritic_conf = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.t_critic_dir)
        self.best_tcritic_weights = os.path.join(sys.path[0], self.res_dir, self.best_dir, self.t_critic_weights)

        self.dirs = [self.res_dir]

    def __str__(self):
        return "\n".join([
                           f"self.total_time_steps = {self.total_time_steps}",
                           f"self.sample_rate = {self.sample_rate}",
                           f"self.episode_sim_time = {self.episode_sim_time}",
                           f"self.steps_per_episode = {self.steps_per_episode}",
                           f"self.number_of_episodes = {self.number_of_episodes}",
                           f"self.gamma = {self.gamma}",
                           f"self.critic_lr = {self.critic_lr}",
                           f"self.actor_lr = {self.actor_lr}",
                           f"self.std_dev = {self.std_dev}",
                           f"self.tau = {self.tau}",
                           f"self.batch_size = {self.batch_size}",
                           f"self.buffer_size = {self.buffer_size}",
                           f"self.show_env = {self.show_env}",
                           f"self.method = {self.method}",
        ])