import random
import numpy as np

from src import config

class Platoon:
    def __init__(self, length, config):
        """Initialize the platoon

        Args:
            length (int): the length of the platoon. Note that the leader parameters
                          such as tau and a_i are passed into the first vehicle 'FOLLOWER'
            config (Config): the global configuration class
        """
        self.config = config
        self.length = length
        self.followers = []
        self.front_accel = random.uniform(-self.config.pl_leader_ia, self.config.pl_leader_ia)

        if config.model == config.modelA:
            self.state_lbs = {0: "ep", 1 : "ev", 2 : "a"}   
            self.num_actions = 1
            self.num_states = 3
            self.pl_leader_tau = self.config.pl_leader_tau
            for i in range(0, length):
                if i == 0:
                    self.followers.append(Vehicle(i, self.config, 
                                          a_lead=self.front_accel, num_states=self.num_states,
                                          num_actions=self.num_actions))
                else:
                    self.followers.append(Vehicle(i, self.config, num_states=self.num_states,
                                          num_actions=self.num_actions)) # do not chain tau or excel throughout states

        elif config.model == config.modelB:
            self.state_lbs = {0: "ep", 1 : "ev", 2 : "a", 3 : "a-1"}
            self.front_exog = random.uniform(-self.config.reset_max_u, self.config.reset_max_u)
            self.pl_leader_tau = self.config.pl_leader_tau

            self.num_actions = 1
            self.num_states = 4
            for i in range(0, length):
                if i == 0:
                    self.followers.append(Vehicle(i, self.config, self.pl_leader_tau, self.front_accel,
                                                  num_states=self.num_states,
                                                  num_actions=self.num_actions))
                else:
                    self.followers.append(Vehicle(i, self.config, self.followers[i-1].tau, self.followers[i-1].x[2],
                                                  num_states=self.num_states, num_actions=self.num_actions)) # chain tau and accel in state
    
    def render(self):
        output = ""
        for f in self.followers:
            output += f"{f.render(str_form=True)} <~ "
        print(output, end="\r", flush=True)
    
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
            follower = self.followers[i]

            if self.config.model == self.config.modelA:
                if i == 0:
                    exog = self.front_accel
                else:
                    exog = self.followers[i-1].x[2] # leading vehicle accel is the exogenous info for model A

            elif self.config.model == self.config.modelB:
                if i == 0:                    
                    exog = self.front_exog
                else:
                    exog = self.followers[i-1].u # leading vehicle input is exogenous for model B

            f_state, f_reward, terminal = follower.step(action, exog)
            states.append(f_state)
            rewards.append(f_reward)

            if terminal:
                break
        
        reward = self.get_reward(states, rewards) * self.config.re_scalar
        return states, reward, terminal
    
    def get_reward(self, states, rewards):
        """Calculates the platoons reward

        Args:
            states (list): the list of the states for all followers in platoon
            rewards (list): the list of rewards for all followers in platoon

        Returns:
            float : the reward of the platoon
        """
        reward = (1/self.length)*sum(rewards)
        return reward

    def reset(self):
        states = []
        self.front_accel = random.uniform(-self.config.pl_leader_ia,self.config.pl_leader_ia)

        for i in range(len(self.followers)):
            if self.config.model == self.config.modelA:
                follower_st = self.followers[i].reset()

            elif self.config.model == self.config.modelB:
                self.front_exog = random.uniform(-self.config.reset_max_u, self.config.reset_max_u)
                if i == 0:
                    follower_st = self.followers[i].reset(self.front_accel)
                else:
                    follower_st = self.followers[i].reset(self.followers[i-1].x[2])

            states.append(follower_st)
        
        return states


class Vehicle:
    """Vehicle class based on constant time headway modeling
            Attributes:
                h           (float)    : the desired time gap (s)
                idx         (int)      : the integer id of the vehicle
                T           (float)    : sample rate
                tau         (float)    : vehicle accel dynamics coefficient
                tau_lead    (float)    : leading vehicle accell dynamics coeff
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
    def __init__(self, idx, config: config.Config, tau_lead=None, a_lead = None, num_states = None, num_actions = None):
        """constructor - r, h, L and tau referenced from 
                         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7146426/
                         a, b taken from https://www.merl.com/publications/docs/TR2019-142.pdf     
            Arguments:
                idx (integer) : the vehicle's index in the platoon
                config (config.Config) : the configuration class
                tau_lead (float, optional) : the vehicle dynamics coefficent of the leading vehicle in the platoon
                a_lead (float, optional) : the acceleration of the leading vehicle in the platoon 
        """
        self.config = config
        self.idx = idx
        self.h = self.config.timegap
        self.T = self.config.sample_rate
        self.tau = self.config.dyn_coeff
        self.tau_lead = tau_lead
        self.a = self.config.reward_ev_coeff
        self.b = self.config.reward_u_coeff

        self.max_ep = self.config.max_ep # max ep error before environment returns terminate = True
        self.reset_ep_max = self.config.reset_ep_max # used as the max +/- value when generating new ep on reset
        self.reset_max_ev = self.config.reset_max_ev  # max +/- value when generating new ev on reset
        self.reset_max_a = self.config.reset_max_a # max +/- value for accel on reset

        self.reward = 0
        self.u = 0 # the input to the system
        self.exog = 0

        self.action_high =  self.config.action_high
        self.action_low  = self.config.action_low 

        self.num_states = num_states
        self.num_actions = num_actions

        if self.config.model == self.config.modelA:         
            self.reset() # initializes the states randomly

            if self.config.method == self.config.euler:
                self.A = np.array([[1, self.T, -self.T*self.h           ],
                                [0, 1     , -self.T                  ],
                                [0, 0     , 1 - self.T * (1/self.tau)]])
                self.B = np.array( [0, 0,      self.T*(1/self.tau)])
                self.C = np.array( [0, self.T, 0])

            elif self.config.method == self.config.exact:
                e = np.exp(-self.T/self.tau)
                self.A = np.array([[1, self.T, - self.h * self.tau 
                                               + self.h * self.tau * e
                                               - self.tau * self.T
                                               + self.tau**2 
                                               - (self.tau**2) * e],
                                    [0,   1   ,    -self.tau + self.tau * e],
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

        elif self.config.model == self.config.modelB:
            e = np.exp(-self.T/self.tau)
            e_lead = np.exp(-self.T/self.tau_lead)
            self.num_states = 4
            self.num_actions = 1

            self.reset(a_lead) # initializes the states randomly

            A_13 = - self.h * self.tau + self.h * self.tau * e \
                   - self.tau * self.T + self.tau**2 - (self.tau ** 2) * e

            A_14 =   self.tau_lead * self.T - self.tau_lead ** 2 \
                   + (self.tau_lead ** 2) * e_lead
            
            A_23 = -self.tau + self.tau * e 

            A_24 =  self.tau_lead - self.tau_lead * e_lead

            self.A = np.array([[1, self.T, A_13,   A_14],
                               [0, 1     , A_23,   A_24],
                               [0, 0     , e   ,   0   ],
                               [0, 0     , 0   , e_lead]])
            
            B_11 = - self.h * self.T + self.h * self.tau * e \
                   - self.h * self.tau - (self.T ** 2) / 2 + self.tau * self.T \
                   + (self.tau ** 2) * e - self.tau ** 2
            
            B_21 = - self.T - self.tau * e + self.tau 

            self.B = np.array([B_11, 
                               B_21,
                               -e + 1,
                               0])

            C_11 =    (self.T ** 2)/2 - self.tau_lead * self.T \
                    - (self.tau_lead ** 2) * e_lead + self.tau_lead ** 2
            C_21 =    self.T + self.tau_lead * e_lead - self.tau_lead

            self.C = np.array([C_11,
                               C_21,
                               0,
                               - e_lead + 1])

    def render(self, str_form=False):
        output = f"|x: {np.round(self.x, 3)}, r: {round(self.reward, 3)}, u: {round(self.u, 3)}, exog: {round(self.exog, 3)}|"
        if str_form == True:
            return output
        else:
            print(output, end='\r', flush=True)
    
    def step(self, u, exog_info):
        """advances the vehicle model by one timestep

        Args:
            u (float): the action to take
            exog_info (float): the exogenous information given to the vehicle
        """
        terminal = False
        self.u = u[0] 
        self.exog = exog_info 
        if abs(self.x[0]) > self.max_ep:
            terminal = True
            self.reward = 1000
            self.x = self.A.dot(self.x) + self.B.dot(u[0]) + self.C.dot(exog_info)

        else:
            self.reward = self.x[0]**2 + self.a*(self.x[1])**2 + self.b*(u[0])**2
            self.x = self.A.dot(self.x) + self.B.dot(u[0]) + self.C.dot(exog_info)

        return self.x, -self.reward, terminal
    
    def reset(self, a_lead=None):
        """reset the vehicle environment

        Args:
            a_lead (float, optional): used in models where acceleration is chained throughout the vehicle state. Defaults to None.
        """
        self.u = 0
        self.exog_info = 0

        if self.config.model == self.config.modelA:
            self.x =  [random.uniform(-self.reset_ep_max, self.reset_ep_max), # intial gap error (m)
                       random.uniform(-self.reset_max_ev, self.reset_max_ev), # initial velocity error
                       random.uniform(-self.reset_max_a,  self.reset_max_a)]   # initial accel of this vehicle

        elif self.config.model == self.config.modelB:
            self.x = [random.uniform(-self.reset_ep_max, self.reset_ep_max), # intial gap error (m)
                      random.uniform(-self.reset_max_ev, self.reset_max_ev), # initial velocity error
                      random.uniform(-self.reset_max_a,  self.reset_max_a), # initial accel of this vehicle
                      a_lead]   # initial accel of leading vehicle

        return(self.x)
    
    def set_state(self, state):
        self.x = state


if __name__=="__main__":
    conf = config.Config()
    pl = Platoon(2, conf)
    # v = Vehicle(1, conf)
    pl.render()
    print("\npre reset")
    print(pl.step([[2.5], [-1.5]]))
    pl.render()
    print("\npost")
    print(pl.reset())

    


    