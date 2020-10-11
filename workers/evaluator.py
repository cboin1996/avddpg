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

def run(conf=None, actor=None, path_timestamp=None, out=None, step_bound=None, const_bound=None, ramp_bound=None):
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress matpltlib warning related to subplot
    
    if conf is None:
        print("Creating new configuration instance from config.py.")
        conf = config.Config()
        conf.step_in = float(step_bound)
        conf.const_in = float(const_bound)
        conf.ramp_bound = float(ramp_bound)
    
    if path_timestamp is None:
        model_parent_dir = Config.best_root
    else:
        model_parent_dir = path_timestamp
    
    env = environment.Platoon(conf.pl_size, conf)
    if actor is None:
        actor = tf.keras.models.load_model(conf.best_actor_conf)

    simulation_time = 20
    steps = int(simulation_time/conf.sample_rate)
    
    ou_noise = noise.OUActionNoise(mean=np.zeros(1), std_dev=float(conf.std_dev) 
                                                                   * np.ones(1))

    input_opts = {conf.zerofig_name : [0 for i in range(steps)],
                conf.constfig_name : [conf.const_in for i in range(steps)],
                conf.stepfig_name : [0 if i < (steps/2) else conf.step_in for i in range(steps)],
                conf.rampfig_name : np.linspace(-conf.ramp_bound, conf.ramp_bound, steps)}

    actions = np.zeros((conf.pl_size, env.num_actions))
    pl_states = np.zeros((steps, conf.pl_size, env.num_states))
    pl_inputs = np.zeros((steps, conf.pl_size, 1))

    num_rows = env.num_states + 1
    num_cols = 1
    for typ, input_list in input_opts.items():
        plt.figure(figsize = (4,12))
        states = env.reset()
        

        for i in range(steps):
            for k, state in enumerate(states):
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                actions[k] = ddpgagent.policy(actor(state), ou_noise, conf.action_low, conf.action_high)

            states, reward, terminal = env.step(actions)

            pl_states[i] = states
            pl_inputs[i] = actions
            
        for i in range(conf.pl_size): # for each follower's states in the platoon states
            for j in range(env.num_states):
                plt.subplot(num_rows, num_cols, j+1)
                plt.plot(pl_states[:,i][:,j], label=f"Vehicle {i}")
                plt.xlabel(f"{conf.sample_rate}s steps (total time of {simulation_time} s)")
                plt.ylabel(f"{env.state_lbs[j]}")
                plt.legend()
            
            plt.subplot(num_rows, num_cols, env.num_states + 1) # create subplot for inputs
            plt.plot(pl_inputs[:, i], label=f"Vehicle {i}")

        plt.plot(input_list, label=f"Platoon input") # overlay platoon input on inputs plot
        plt.xlabel(f"{conf.sample_rate}s steps (total time of {simulation_time} s)")
        plt.ylabel("inputs")
        plt.legend()

        plt.suptitle(f"{typ} input response.")
        plt.tight_layout()

        if out == 'save':
            out_file = os.path.join(model_parent_dir, f"res_{typ}.png")
            print(f"Generated {typ} simulation plot to -> {out_file}")
            plt.savefig(out_file)
        else:
            
            plt.show()
