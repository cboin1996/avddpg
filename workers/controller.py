    
from src import environment
from src import config
import matplotlib.pyplot as plt
import numpy as np
import math
def run():
    conf = config.Config()
    simulation_time = 20 # sim time in seconds
    steps = int(simulation_time/conf.sample_rate)

    env = environment.Vehicle(1, conf)

    print(env)

    high_bound = env.action_high
    low_bound  = env.action_low

    print(f"Total episodes: {conf.number_of_episodes}\nSteps per episode: {conf.steps_per_episode}")

    controller = PID(4.5, 10, 0) # initialize controller
    states = np.zeros((steps, env.num_states)) # states
    input_list = [] # input list for plotting
    state = env.x # first state
    for i in range(steps):
        inp = math.sin(i)
        input_list.append(inp)

        action = controller.control(conf.sample_rate, inp - state[2])
        state, reward, terminal = env.step([action], 0)
        states[i] = state

    plt.plot(states[:, 0], label="ep")
    plt.plot(states[:, 1], label="ev")
    plt.plot(states[:, 2], label="a")
    plt.plot(input_list, label="u")
    plt.xlabel(f"{conf.sample_rate}s steps for total time of {simulation_time} s")
    plt.legend()
    plt.show()

class PID:
    def __init__(self, kp, ki, kd):
        """Default constructor for PID

        Args:
            kp (float): proportional gain
            ki (float): integral gain
            kd (float): differential gain
        """
        self.cumulative_err = 0
        self.delta_err = 0
        self.last_err = 0

        self.kp = kp
        self.ki = ki 
        self.kd = kd
    
    def control(self, dt, error):
        """Apply the PID control to the signal error

        Args:
            dt (integer): the timestep for the control
            error (float): the error

        Returns:
            float: the control output
        """
        self.cumulative_err += error
        self.delta_err = error - self.last_err

        output =   self.kp * error \
                 + self.ki * dt * self.cumulative_err \
                 + (self.kd/dt) * self.delta_err
        self.last_err = error
        return output
