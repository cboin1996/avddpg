import enum
import os, sys
import random
class Config():
    modelA = 'ModelA'
    modelB = 'ModelB'
    model = modelB
    
    res_dir = 'res'
    param_path = "conf.json"

    def __init__(self):
        """Vars that we want to access statically, but also through a simple name space need redundant declaration"""
        self.modelA = self.modelA
        self.modelB = self.modelB
        self.model = self.model
        
        self.res_dir = self.res_dir
        self.param_path = self.param_path

        """Environment"""
        self.pl_size = 1 # the size of the platoon.
        self.pl_leader_reset_a = 0 # max initial acceleration of the platoon leader (used in the calculation for \dot{a_{i-1}}) (bound for uniform, std_dev for normal)
        self.reset_max_u = 0.100 # max initial control input of the platoon leader (used in the calculation for \dot{a_{i-1}}, (bound for uniform, std_dev for normal)

        self.pl_leader_tau = 0.5
        self.exact = 'exact'
        self.euler = 'euler'
        self.method = self.exact

        self.timegap = 1.25
        self.dyn_coeff = 0.5
        self.reward_ev_coeff = 1
        self.reward_u_coeff = 0.10

        self.max_ep = 10
        self.max_ev = 10
        
        self.reset_ep_max = 1.5
        self.reset_max_ev = 1.5
        self.reset_max_a = 0.05 # max accel of a vehicle upon reset

        self.action_high =4.5
        self.action_low = -4.5

        self.re_scalar = 0.1 # reward scale
        self.terminal_reward = 1000

        """Trainer"""
        self.can_terminate = True
        self.random_seed = 2
        self.normal = 'normal' 
        self.uniform = 'uniform'
        self.rand_gen = self.normal # which type of random numbers to use.

        self.total_time_steps = 1000000

        self.sample_rate = 0.1
        self.episode_sim_time = 60 # simulation time for a training episode
        self.steps_per_episode = int(self.episode_sim_time/self.sample_rate)

        self.number_of_episodes = int(self.total_time_steps/self.steps_per_episode)
        
        self.gamma = 0.99 # Discount factor for future rewards

        # Learning rate for actor-critic models
        self.critic_lr = 0.001
        self.actor_lr = 0.0001
        self.std_dev = 0.02 # orhnstein gaussian noise standard dev
        self.theta = 0.15 # orhstein theta
        self.ou_dt = 1e-2 # ornstein dt
        self.tau = 0.001 # target network update coeff

        self.batch_size=64
        self.buffer_size=100000
        self.show_env=False
        """Directories"""

        self.actor_fname = 'actor.h5'
        self.actor_picname = 'actor.png'
        self.actor_weights = 'actor_weights.h5'
        self.critic_fname = 'critic.h5'
        self.critic_picname = 'critic.png'
        self.critic_weights = 'critic_weights.h5'
        self.t_actor_fname = 'target_actor.h5'
        self.t_actor_picname = 'target_actor.png'
        self.t_actor_weights = 'target_actor_weights.h5'
        self.t_critic_fname = 'target_critic.h5'
        self.t_critic_picname = 'target_critic.png'
        self.t_critic_weights = 'target_critic_weights.h5'

        self.fig_path = "reward_curve.png"
        
        self.zerofig_name = "zero"
        self.guasfig_name = "guassian"
        self.stepfig_name = "step"
        self.rampfig_name = "ramp"

        self.dirs = [self.res_dir]



if __name__=="__main__":
    import util
    conf = Config()

    util.config_writer("conf.json", conf)

