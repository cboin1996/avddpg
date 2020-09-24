import random
import numpy as np
from src import config

class Platoon:
    def __init__(self, length, T, config):
        self.length = length
        self.T = T
        self.vehicles = [Vehicle(i, T, config) for i in range(0,length)]
        self.idx = 0
        self.front_accel = random.uniform(-3,3)
    
    def __str__(self):
        return str([str(v) for v in self.vehicles])
    
    def step(self, actions):
        """Advances the environment one step

        Args:
            actions (list): list of actions from the DNN model

        Returns:
            list, float : a list of states calculated for the platoon,
                          the platoon reward
        """
        states = []
        rewards = []
        for i, action in enumerate(actions):
            vehicle = self.vehicles[i]
            if i < len(self.vehicles):
                a_lead = self.vehicles[i+1].x[2]
            else:
                a_lead = self.front_accel
            v_state, v_reward = vehicle.step(action, a_lead)
            states.append(v_state)
            rewards.append(v_reward)

        reward = self.get_reward(states, rewards)
        return states, reward
    
    def get_reward(self, states, rewards):
        """Calculates the platoons reward

        Args:
            states (list): the list of the states for all vehicles in platoon
            rewards (list): the list of rewards for all vehicles in platoon

        Returns:
            float : the reward of the platoon
        """
        reward = -(1/np.linalg.norm(states))*sum(rewards)
        return reward

    def reset(self):
        self.front_accel = random.uniform(-3,3)


class Vehicle:
    """Vehicle class based on constant time headway modeling
    """
    def __init__(self, idx, config):
        """constructor - r, h, L and tau referenced from 
                         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7146426/
                         a, b taken from https://www.merl.com/publications/docs/TR2019-142.pdf


        Attributes:
            r           (float)    : the constant standstill distance for a vehicle (m)
            h           (float)    : the desired time gap (s)
            L           (float)    : the length of the vehicle (m)
            idx         (int)      : the integer id of the vehicle
            T           (float)    : sample rate
            tau         (float)    : vehicle accel dynamics coefficient
            num_states  (int)      : number of states in the model
            num_actions (int)      : number of actions in the model
            a           (float)    : plant stability constant
            b           (float)    : string stability constant
            x           (np.array) : the state of the system (control error ep, 
                                     control error ev, vehicle acceleration a)
            A           (np.array) : system matrix  
            B           (np.array) : system matrix
            C           (np.array) : system matrix
            config      (Config)   : configuration class
        """
        self.r = 8
        self.h = 1.25
        self.L = round(random.uniform(2,3), 2)
        self.idx = idx
        self.T = config.sample_rate
        self.tau = 0.5
        self.num_states = 3
        self.num_actions = 1
        self.a = 1
        self.b = 0.10

        self.i_ep = 2.5
        self.i_ev = 2.5
        self.i_alead = 0.0

        self.max_ep = 15 # max ep error before environment returns terminate = True
        self.reset_ep_max = 5 # used as the max +/- value when generating new ep on reset
        self.reset_max_ev = 3 # max +/- value when generating new ev on reset

        self.x = [self.i_ep, # intial gap error (m)
                  self.i_ev, # initial velocity error
                  self.i_alead]   # initial accel of leading vehicle
        self.reward = 0
        self.u = 0 # the input to the system (vehicle acceleration)

        self.action_high =  4.5
        self.action_low  = -4.5

        if config.method == config.euler:
            self.A = np.array([[1, self.T, -self.T*self.h           ],
                               [0, 1     , -self.T                  ],
                               [0, 0     , 1 - self.T * (1/self.tau)]])
            self.B = np.array( [0, 0,      self.T*(1/self.tau)])
            self.C = np.array( [0, self.T, 0])

        elif config.method == config.exact:
            e = np.exp(-self.T/self.tau)
            self.A = np.array([[1, self.T, - self.h * self.tau 
                                           + self.h * self.tau * e
                                           - self.tau * self.T
                                           + self.tau**2 
                                           - (self.tau**2) * e],
                               [0,   1   , -self.tau + self.tau * e],
                               [0,   0   , e]])

            B_11 = - self.h * self.T \
                   - self.h * self.tau * e\
                   + self.h * self.tau \
                   - (self.T**2)/2 \
                   + self.tau * self.T \
                   + (self.tau ** 2) * e - self.tau**2
            
            B_21 = - self.T - self.tau * e + self.tau 
            B_31 = - e + 1
            self.B = np.array([B_11, B_21, B_31]) 
            self.C = np.array([(self.T**2)/2, self.T, 0])

    def __str__(self):
        return "\n".join([
                            f"self.r = {self.r}",
                            f"self.h = {self.h}",
                            f"self.L = {self.L}",
                            f"self.idx = {self.idx}",
                            f"self.T = {self.T}",
                            f"self.tau = {self.tau }",
                            f"self.num_states = {self.num_states}",
                            f"self.num_actions = {self.num_actions}",
                            f"self.a = {self.a}",
                            f"self.b = {self.b}",
                            f"self.i_ep = {self.i_ep}",
                            f"self.i_ev = {self.i_ev}",
                            f"self.i_alead = {self.i_alead}",
                            f"self.x = {self.x}",
                            f"self.action_high = {self.action_high}",
                            f"self.action_low = {self.action_low}",
                            f"self.max_ep = {self.max_ep}",
                            f"self.reset_ep_max = {self.reset_ep_max}",
                            f"self.reset_max_ev = {self.reset_max_ev}"
                        ])
    
    def render(self):
        return print(" ".join([
                            f"x: {self.x}, reward: {self.reward}, u: {self.u}"
                        ]), end="\r", flush=True)
    
    def step(self, u, a_lead):
        """advances the vehicle model by one timestep

        Args:
            u (float): the action to take
            a_lead (float): the leading vehicles acceleration
        """
        terminal = False
        self.u = u[0]

        if abs(self.x[0]) > self.max_ep:
            terminal = True
            self.reward = 1000
            self.x = self.A.dot(self.x) + self.B.dot(u[0]) + self.C.dot(a_lead)
        else:
            self.reward = self.x[0]**2 + self.a*(self.x[1])**2 + self.b*(u[0])**2
            self.x = self.A.dot(self.x) + self.B.dot(u[0]) + self.C.dot(a_lead)
        
        return self.x, -self.reward, terminal
    
    def reset(self):
        self.x =  [random.uniform(-self.reset_ep_max, self.reset_ep_max), # intial gap error (m)
                   random.uniform(-self.reset_max_ev, self.reset_max_ev), # initial velocity error
                   self.i_alead]   # initial accel of leading vehicle
        return(self.x)
    
    def set_state(self, state):
        self.x = state


if __name__=="__main__":
    v = Vehicle(0, config.Config)
    print(v.x)
    v.step(1.5, 0)
    print(v.x)
    v.step(1.5, 0)
    print(v.x)
    v.step(3, 0)
    print(v.x)
    v.reset()
    print(v.x)
    # platoon = Platoon(5, 1e-3)
    # print(str(platoon))

    


    