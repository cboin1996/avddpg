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

import warnings

def run(conf=None, actor=None, path_timestamp=None, out=None, step_bound=None, const_bound=None, ramp_bound=None, root_path=None):
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress matpltlib warning related to subplot
    
    if conf is None:
        conf_path = os.path.join(root_path, config.Config.param_path)
        print(f"Loading configuration instance from {conf_path}")
        conf = util.config_loader(conf_path)
    
    if path_timestamp is None:
        model_parent_dir = root_path
    else:
        model_parent_dir = path_timestamp
    
    env = environment.Platoon(conf.pl_size, conf, rand_states=False) # do not use random states here, for consistency across training sessions

    if step_bound is None or const_bound is None or ramp_bound is None:
        print(f"Setting step, const and ramp defaults all to {env.pl_leader_max_exog}")
        step_bound = env.pl_leader_max_exog
        const_bound = env.pl_leader_max_exog
        ramp_bound = env.pl_leader_max_exog

    if actor is None:
        actor = tf.keras.models.load_model(os.path.join(root_path, conf.actor_fname))

    simulation_time = 20
    steps = int(simulation_time/conf.sample_rate)
    
    ou_noise = noise.OUActionNoise(mean=np.zeros(1), std_dev=float(conf.std_dev) 
                                                                   * np.ones(1))

    input_opts = {conf.zerofig_name : [0 for _ in range(steps)],
                conf.constfig_name : [const_bound for _ in range(steps)],
                conf.stepfig_name : [0 if i < (steps/2) else step_bound for i in range(steps)],
                conf.rampfig_name : np.linspace(-1*ramp_bound, ramp_bound, steps)}

    actions = np.zeros((conf.pl_size, env.num_actions))
    pl_states = np.zeros((steps, conf.pl_size, env.num_states))
    pl_inputs = np.zeros((steps, conf.pl_size, 1))

    num_rows = env.num_states + 1
    num_cols = 1
    cum_reward = 0
    for typ, input_list in input_opts.items():
        plt.figure(figsize = (4,12))
        states = env.reset()
        
        for i in range(steps):
            for k, state in enumerate(states):
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                actions[k] = ddpgagent.policy(actor(state), ou_noise, conf.action_low, conf.action_high)

            states, reward, terminal = env.step(actions, input_list[i])
            cum_reward += reward
            pl_states[i] = states
            pl_inputs[i] = actions
            
        for i in range(conf.pl_size): # for each follower's states in the platoon states
            for j in range(env.num_states):
                plt.subplot(num_rows, num_cols, j+1)
                plt.plot(pl_states[:,i][:,j], label=f"Vehicle {i+1}")
                plt.xlabel(f"{conf.sample_rate}s steps (total time of {simulation_time} s)")
                plt.ylabel(f"{env.state_lbs[j]}")
                plt.legend()
            
            plt.subplot(num_rows, num_cols, env.num_states + 1) # create subplot for inputs
            plt.plot(pl_inputs[:, i], label=f"Vehicle {i+1}")

        plt.plot(input_list, label=f"Platoon leader's {env.exog_lbl}") # overlay platoon leaders transmitted data
        plt.xlabel(f"{conf.sample_rate}s steps (total time of {simulation_time} s)")
        plt.ylabel("u")
        plt.legend()

        plt.suptitle(f"{conf.model} {typ} input response\n with cumulative reward of {round(cum_reward, 2)}")
        plt.tight_layout()

        if out == 'save':
            out_file = os.path.join(model_parent_dir, f"res_{typ}.png")
            print(f"Generated {typ} simulation plot to -> {out_file}")
            plt.savefig(out_file)
        else:
            
            plt.show()
