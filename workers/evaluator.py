import tensorflow as tf
import numpy as np
from src import config, noise, replaybuffer, environment, util
from agent import model, ddpgagent
import gym
import matplotlib.pyplot as plt
import h5py
import math

def run():
    conf = config.Config()
    print(conf)
    print("")
    env = environment.Vehicle(1, conf)
    print(env)
    actor = tf.keras.models.load_model(conf.best_actor_conf)

    simulation_time = 20
    steps = int(simulation_time/conf.sample_rate)
    env.set_state([5,2.5, -2.5])
    state = env.x

    ou_noise = noise.OUActionNoise(mean=np.zeros(1), std_dev=float(conf.std_dev) 
                                                                   * np.ones(1))

    states = np.zeros((steps, env.num_states)) # states
    # input_list = np.linspace(-2.5, 2.5, steps)
    # input_list = [0 if i < (steps/2) else 2.5 for i in range(steps)]
    input_list = [0 for i in range(steps)]
    for i in range(steps):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = ddpgagent.policy(actor(state), ou_noise, env.action_low, env.action_high)

        state, reward, terminal = env.step(action, input_list[i])

        states[i] = state
    
    plt.plot(states[:, 0], label="ep")
    plt.plot(states[:, 1], label="ev")
    plt.plot(states[:, 2], label="a")
    plt.plot(input_list, label="a_lead")
    plt.xlabel(f"{conf.sample_rate}s steps for total time of {simulation_time} s")
    plt.legend()
    plt.show()

