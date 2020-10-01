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

def run(conf=None, actor=None, path_timestamp=None):
    if conf is None:
        print("Creating new configuration instance from config.py.")
        conf = config.Config()
    
    if path_timestamp is None:
        model_parent_dir = Config.best_root
    else:
        model_parent_dir = path_timestamp
    
    env = environment.Vehicle(1, conf)
    if actor is None:
        actor = tf.keras.models.load_model(conf.best_actor_conf)

    simulation_time = 20
    steps = int(simulation_time/conf.sample_rate)
    
    ou_noise = noise.OUActionNoise(mean=np.zeros(1), std_dev=float(conf.std_dev) 
                                                                   * np.ones(1))

    input_opts = {conf.zerofig_name : [0 for i in range(steps)],
                  conf.constfig_name : [2.5 for i in range(steps)],
                  conf.stepfig_name : [0 if i < (steps/2) else 2.5 for i in range(steps)],
                  conf.rampfig_name : np.linspace(-2.5, 2.5, steps)}

    for typ, input_list in input_opts.items():
        env.reset()
        env.set_state([5,2.5, -2.5])
        state = env.x
        states = np.zeros((steps, conf.num_states))
        for i in range(steps):
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = ddpgagent.policy(actor(state), ou_noise, env.action_low, env.action_high)

            state, reward, terminal = env.step(action, input_list[i])

            states[i] = state

        plt.figure()
        plt.plot(states[:, 0], label="ep")
        plt.plot(states[:, 1], label="ev")
        plt.plot(states[:, 2], label="a")
        plt.plot(input_list, label="a_lead")
        plt.xlabel(f"{typ} input response for {conf.sample_rate}s steps (total time of {simulation_time} s)")
        plt.legend()
        out_file = os.path.join(model_parent_dir, f"res_{typ}.png")
        print(f"Generated {typ} simulation plot to -> {out_file}")
        plt.savefig(out_file)
