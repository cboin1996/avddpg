import enum
import os, sys
import random
class Config():
    modelA = 'ModelA'
    modelB = 'ModelB'
    model = modelB
    
    res_dir = os.path.join('.outputs')
    param_path = "conf.json"

    def __init__(self):
        """Vars that we want to access statically, but also through a simple name space need redundant declaration"""
        self.modelA = self.modelA
        self.modelB = self.modelB
        self.model = self.model
        self.dcntrl = "decentralized"
        self.cntrl = "centralized"
        self.hfrl = "horizontal federated"
        self.vfrl = "vertical federated"
        self.nofrl = "normal"
        self.fed_method = self.hfrl
        self.framework = self.dcntrl
        self.fed_enabled = (self.fed_method == self.hfrl or self.fed_method == self.vfrl) and (self.framework == self.dcntrl)
        self.res_dir = self.res_dir
        self.report_dir = "reports"
        self.param_path = self.param_path

        """Environment"""
        self.num_platoons = 2 # the number of platoons for training and simulation
        self.pl_size = 1 # the number of following vehicles in the platoon.
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

        self.max_ep = 20
        self.max_ev = 10
        
        self.reset_ep_max = 1.5
        self.reset_max_ev = 1.5
        self.reset_max_a = 0.05 # max accel of a vehicle upon reset

        self.action_high =2.5
        self.action_low = -2.5

        self.re_scalar = 0.1 # reward scale
        self.terminal_reward = 1000

        """Trainer"""
        self.can_terminate = True
        self.random_seed = 3
        self.evaluation_seed = 6

        self.normal = 'normal' 
        self.uniform = 'uniform'
        self.rand_gen = self.normal # which type of random numbers to use.

        self.total_time_steps = 1000

        self.sample_rate = 0.1
        self.episode_sim_time = 60 # simulation time for a training episode
        self.steps_per_episode = int(self.episode_sim_time/self.sample_rate)

        self.number_of_episodes = int(self.total_time_steps/self.steps_per_episode)
        
        self.gamma = 0.99 # Discount factor for future rewards

        self.centrl_hidd_mult = 1.2
        
        # Learning rate for actor-critic models
        self.critic_lr = 0.0005
        self.actor_lr = 0.00005
        self.std_dev = 0.02 # orhnstein gaussian noise standard dev
        self.theta = 0.15 # orhstein theta
        self.ou_dt = 1e-2 # ornstein dt
        self.tau = 0.001 # target network update coeff

        self.batch_size=64
        self.buffer_size=100000
        self.show_env=False

        """Models"""
        self.actor_layer1_size=256
        self.actor_layer2_size=128

        self.critic_layer1_size=256
        self.critic_layer2_size=128
        """Directories"""
        self.img_tag = "%s_%s"
        self.actor_fname = f'actor{self.img_tag}.h5'
        self.actor_picname = f'actor{self.img_tag}.png'
        self.actor_weights = f'actor_weights{self.img_tag}.h5'
        self.critic_fname = f'critic{self.img_tag}.h5'
        self.critic_picname = f'critic{self.img_tag}.png'
        self.critic_weights = f'critic_weights{self.img_tag}.h5'
        self.t_actor_fname = f'target_actor{self.img_tag}.h5'
        self.t_actor_picname = f'target_actor{self.img_tag}.png'
        self.t_actor_weights = f'target_actor_weights{self.img_tag}.h5'
        self.t_critic_fname = f'target_critic{self.img_tag}.h5'
        self.t_critic_picname = f'target_critic{self.img_tag}.png'
        self.t_critic_weights = f'target_critic_weights{self.img_tag}.h5'

        self.pl_tag = "_p%s"
        self.fig_path = f"reward_curve{self.pl_tag}.png"
        
        self.zerofig_name = "zero"
        self.guasfig_name = "guassian"
        self.stepfig_name = "step"
        self.rampfig_name = "ramp"

        self.dirs = [self.res_dir, self.report_dir]

        """Logging"""
        self.log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        self.log_date_fmt = "%y-%m-%d %H:%M:%S"

        """Reporting"""
        self.pl_rew_for_simulation = 0
        self.pl_rews_for_simulations = []
        self.index_col = "timestamp"
        self.timestamp = None
        self.drop_keys_in_report = ["modelA", "modelB", "dcntrl", "cntrl", "hfrl", "vfrl", "nofrl", "res_dir", "report_dir", "param_path", "euler",
                                    "exact", "normal", "uniform", "show_env", "actor_fname", "actor_picname", "actor_weights", "critic_fname", "critic_picname",
                                    "critic_weights", "t_actor_fname", "t_actor_picname", "t_actor_weights", "t_critic_fname", "t_critic_picname", 
                                    "t_critic_weights", "fig_path", "zerofig_name", "guasfig_name", "stepfig_name", "rampfig_name", "dirs",
                                    "log_format", "log_date_fmt", "drop_keys_in_report", "index_col", "param_descs", "img_tag", "pl_rews_for_simulations"]

        self.param_descs = {"timestamp" : "The time at which the experiment was run",
                            "model" : "Whether a 3 (ModelA) of 4 (ModelB) state model",
                            "fed_method" : "Type of federated learning used",
                            "framework" : "Decentralized or centralized",
                            "fed_enabled" : "Whether FRL is enabled",
                            "num_platoons" : "The number of platoons",
                            "pl_size" : "The size of the platoon",
                            "pl_leader_reset_a" : "The mean value of the platoon leader acceleration upon environment reset",
                            "reset_max_u" : "The mean value of the platoon leader control input upon environment reset",
                            "pl_leader_tau" : "Vehicle dynamics coefficient",
                            "method" : "Exact or euler discretization",
                            "timegap" : "Constant time headway CACC time gap",
                            "dyn_coeff" : "Dynamic coefficients for the following vehicles",
                            "reward_ev_coeff" : "Coefficient of velocity difference in reward equation",
                            "reward_u_coeff" : "Coefficient of control input in reward equation",
                            "max_ep" : "Maximum position error in followers before episode termination",
                            "max_ev" : "Maximum position error in followers before episode termination",
                            "reset_ep_max" : "Maximum position error in followers upon environment reset",
                            "reset_max_ev" : "Maximum velocity error in followers upon environment reset",
                            "reset_max_a" : "Maximum acceleration upon environment reset",
                            "action_high" : "Upper bound on action space for the environment",
                            "action_low" : "Lower bound on action space for the environment",
                            "re_scalar" : "Reward scaling coefficient",
                            "terminal_reward" : "Reward assigned upon early termination",
                            "can_terminate" : "Whether the environment is allowed to terminate early",
                            "random_seed" : "Seed for the experiment across all python libraries",
                            "evaluation_seed" : "Seed used during final simulation",
                            "rand_gen" : "Either uniform or normal for random number generation",
                            "total_time_steps" : "The number of timesteps to train on",
                            "sample_rate" : "The sample rate of the system",
                            "episode_sim_time" : "The time in seconds to run simulations for",
                            "steps_per_episode" : "The number of steps per episode",
                            "number_of_episodes" : "The total number of episodes for training",
                            "gamma" : "The discounted reward coefficient",
                            "centrl_hidd_mult" : "Multiplier for number of nodes across hidden layers",
                            "critic_lr" : "Learning rate for critic network",
                            "actor_lr" : "Learning rate for actor network",
                            "std_dev" : "Standard deviation used in OU noise",
                            "theta" : "Theta value for OU noise",
                            "ou_dt" : "Sample rate for OU noise",
                            "tau" : "Target network update parameter",
                            "batch_size" : "Batch size for sampling replay buffer",
                            "buffer_size" : "Size of the replay buffer",
                            "pl_rews_for_simulations" : "Stores all platoon's rewards after the final simulation",
                            "pl_rew_for_simulation" : "The average of the platoon rewards after the final simulation",
                            "actor_layer1_size" : "Number of nodes in actor's first hidden layer",
                            "actor_layer2_size" : "Number of nodes in actor's second hidden layer",
                            "critic_layer1_size" : "Number of nodes in critic's first hidden layer",
                            "critic_layer2_size" : "Number of nodes in critics's second hidden layer"}

if __name__=="__main__":
    import util
    conf = Config()

    util.config_writer("conf.json", conf)

