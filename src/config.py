import enum
import os, sys
import random
class Config():
    modelA = 'modelA'
    modelB = 'modelB'
    model = modelB
    
    res_dir = 'res'
    best_dir = 'best_' + model
    param_path = "conf.json"
    best_root = os.path.join(sys.path[0], res_dir, best_dir)
    best_param_path = os.path.join(best_root, param_path)


    def __init__(self):
        """Environment"""
        self.pl_size = 2 # the size of the platoon.
        self.pl_leader_ia = 1 # max initial acceleration of the platoon leader
        self.pl_leader_tau = 0.5
        self.exact = 'exact'
        self.euler = 'euler'
        self.method = self.exact

        self.timegap = 1.25
        self.dyn_coeff = 0.5
        self.reward_ev_coeff = 1
        self.reward_u_coeff = 0.10

        self.max_ep = 15
        
        self.reset_ep_max = 5
        self.reset_max_ev = 3
        self.reset_max_a = 1
        self.reset_max_u = 0.5

        self.action_high = 4.5
        self.action_low = -4.5

        self.re_scalar = 0.1 # reward scale

        """Trainer"""
        self.total_time_steps = 500000

        self.sample_rate = 0.1
        self.episode_sim_time = 60 # simulation time for a training episode
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
        self.show_env=True



        """Directories"""

        self.actor_fname = 'actor.h5'
        self.actor_weights = 'actor_weights.h5'
        self.critic_fname = 'critic.h5'
        self.critic_weights = 'critic_weights.h5'
        self.t_actor_fname = 'target_actor.h5'
        self.t_actor_weights = 'target_actor_weights.h5'
        self.t_critic_fname = 'target_critic.h5'
        self.t_critic_weights = 'target_critic_weights.h5'

        self.fig_path = "reward_curve.png"
        
        self.zerofig_name = "Zero"
        self.constfig_name = "Constant"
        self.stepfig_name = "Step"
        self.rampfig_name = "Ramp"

        self.best_actor_conf = os.path.join(self.best_root, self.actor_fname)
        self.best_actor_weights = os.path.join(self.best_root, self.actor_weights)

        self.best_critic_conf = os.path.join(self.best_root, self.critic_fname)
        self.best_critic_weights = os.path.join(self.best_root, self.critic_weights)

        self.best_tactor_conf = os.path.join(self.best_root, self.t_actor_fname)
        self.best_tactor_weights = os.path.join(self.best_root, self.t_actor_weights)

        self.best_tcritic_conf = os.path.join(self.best_root, self.t_critic_fname)
        self.best_tcritic_weights = os.path.join(self.best_root, self.t_critic_weights)

        self.dirs = [self.res_dir]



if __name__=="__main__":
    import util
    conf = Config()

    # util.config_writer("achybreaky.json", conf)
    loaded_conf = util.config_loader("achybreaky.json")
    print(loaded_conf.critic_dir)