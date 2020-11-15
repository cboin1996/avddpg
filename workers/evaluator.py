import tensorflow as tf
import numpy as np
from src import config, noise, replaybuffer, environment, util
from agent import model, ddpgagent
import gym
import matplotlib.pyplot as plt
import h5py
import math
from src.config import Config
import os
import random

import warnings

def run(conf=None, actor=None, path_timestamp=None, out=None, step_bound=None, const_bound=None, ramp_bound=None, root_path=None, seed=True):

    if conf is None:
        conf_path = os.path.join(root_path, config.Config.param_path)
        print(f"Loading configuration instance from {conf_path}")
        conf = util.config_loader(conf_path)
    
    if path_timestamp is None:
        model_parent_dir = root_path
    else:
        model_parent_dir = path_timestamp
    
    if seed:
        np.random.seed(conf.random_seed)
        tf.random.set_seed(conf.random_seed)
        os.environ['PYTHONHASHSEED']=str(conf.random_seed)
        random.seed(conf.random_seed)

    env = environment.Platoon(conf.pl_size, conf, rand_states=False) # do not use random states here, for consistency across evaluation sessions

    if actor is None:
        actor = tf.keras.models.load_model(os.path.join(root_path, conf.actor_fname))

    input_opts = {conf.guasfig_name : [util.get_random_val(conf.rand_gen, conf.reset_max_u, std_dev=conf.reset_max_u, config=conf)
                                        for _ in range(conf.steps_per_episode)]}

    actions = np.zeros((conf.pl_size, env.num_actions))
    pl_states = np.zeros((conf.steps_per_episode, conf.pl_size, env.num_states))
    pl_inputs = np.zeros((conf.steps_per_episode, conf.pl_size, 1))

    num_rows = env.num_states + 1
    num_cols = 1
    cum_reward = 0
    for typ, input_list in input_opts.items():
        plt.figure(figsize = (4,12))
        states = env.reset()
        
        for i in range(conf.steps_per_episode):
            for k, state in enumerate(states):
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                actions[k] = ddpgagent.policy(actor(state), lbound=conf.action_low, hbound=conf.action_high) # do not use noise in the simulation

            states, reward, terminal = env.step(actions, input_list[i])
            cum_reward += reward
            pl_states[i] = states
            pl_inputs[i] = actions
            
        for i in range(conf.pl_size): # for each follower's states in the platoon states
            for j in range(env.num_states):
                plt.subplot(num_rows, num_cols, j+1)
                plt.plot(pl_states[:,i][:,j], label=f"Vehicle {i+1}")
                plt.xlabel(f"{conf.sample_rate}s steps (total time of {conf.episode_sim_time} s)")
                plt.ylabel(f"{env.state_lbs[j]}")
                plt.legend()
            
            plt.subplot(num_rows, num_cols, env.num_states + 1) # create subplot for inputs
            plt.plot(pl_inputs[:, i], label=f"Vehicle {i+1}")

        plt.plot(input_list, label=f"Platoon leader's {env.exog_lbl}") # overlay platoon leaders transmitted data
        plt.xlabel(f"{conf.sample_rate}s steps (total time of {conf.episode_sim_time} s)")
        plt.ylabel("u")
        plt.legend()

        plt.suptitle(f"{conf.model} {typ} input response\n with cumulative reward of {round(cum_reward, 2)}\n")
        plt.tight_layout()

        if out == 'save':
            out_file = os.path.join(model_parent_dir, f"res_{typ}.png")
            print(f"Generated {typ} simulation plot to -> {out_file}")
            plt.savefig(out_file)
        else:
            
            plt.show()
