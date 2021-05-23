import random
import numpy as np
from src import config, util
import logging

log = logging.getLogger(__name__)

class Platoon:
    def __init__(self, length, config, pl_idx, rand_states=True):
        """Initialize the platoon

        Args:
            length (int): the length of the platoon. Note that the leader parameters
                          such as tau and a_i are passed into the first vehicle 'FOLLOWER'
            config (Config): the global configuration class
            rand_states (Boolean) : whether the platoon will use random states, or the max values from the config on reset.
        """
        self.pl_idx = pl_idx
        log.info(f"=== INITIALIZING PLATOON {self.pl_idx} ===")
        self.config = config
        
        self.length = length
        self.followers = []
        self.front_accel = util.get_random_val(self.config.rand_gen, self.config.pl_leader_reset_a, std_dev=self.config.pl_leader_reset_a, config=self.config)

        self.rand_states = rand_states
        self.state_lbs = {0: "ep", 1 : "ev", 2 : "a", 3 : "a-1"}
        self.exog_lbl = 'u'
        self.front_u = util.get_random_val(self.config.rand_gen, self.config.reset_max_u, std_dev=self.config.reset_max_u, config=self.config)
        self.pl_leader_tau = self.config.pl_leader_tau

        if self.config.framework == self.config.cntrl:
            self.multiplier = self.length
            self.hidden_multiplier = self.config.centrl_hidd_mult
            self.num_models = 1
        else:
            self.multiplier = 1
            self.hidden_multiplier = 1
            self.num_models = self.length

        self.def_num_actions = 1 # useful for plotting
        self.num_actions = self.def_num_actions * self.multiplier

        if self.config.model == self.config.modelA:
            self.def_num_states = 3 # useful for tracking the states numbers in the followers
            self.num_states = self.def_num_states * self.multiplier
        else:
            self.def_num_states = 4
            self.num_states = self.def_num_states * self.multiplier
        
        for i in range(0, length):
            if i == 0:
                self.followers.append(Vehicle(i, self.config, self.pl_leader_tau, self.front_accel,
                                                num_states=self.num_states,
                                                num_actions=self.num_actions, rand_states=self.rand_states))
            else:
                self.followers.append(Vehicle(i, self.config, self.followers[i-1].tau, self.followers[i-1].x[2],
                                                num_states=self.num_states, num_actions=self.num_actions, rand_states=self.rand_states)) # chain tau and accel in state
    
    def render(self):
        output = ""
        for f in self.followers:
            output += f"{f.render(str_form=True)} <~ "
        print(output, end="\r", flush=True)
    


    def step(self, actions, leader_exog=None, debug_mode=False):
        """Advances the environment one step

        Args:
            actions (list): list of actions from the DNN model
            leader_exog (float): the exogenous value for the platoon leader
            debug_mode (bool): whether to run debug mode

        Returns:
            list, float : a list of states calculated for the platoon,
                          the platoon reward
        """
        states = []
        rewards = []
        terminals = []
        for i, action in enumerate(actions):
            follower = self.followers[i]

            if i == 0:                    
                exog = self.front_u if leader_exog == None else leader_exog
            else:
                exog = self.followers[i-1].u 

            f_state, f_reward, f_terminal = follower.step(action, exog, debug_mode)
            states.append(f_state)
            rewards.append(f_reward)
            terminals.append(f_terminal)

        if self.config.framework == self.config.cntrl: 
            states = [list(np.concatenate(states).flat)]
            rewards = [self.get_reward(states, rewards)]

        platoon_done = True if True in terminals else False
        return states, rewards, platoon_done
    
    def get_exogenous_info(self, idx, leader_exog):

        if self.config.model == self.config.modelB:
            if idx == 0:                    
                exog = self.front_u if leader_exog == None else leader_exog
            else:
                exog = self.followers[idx-1].u # model B uses the control input, u as exogenous

        elif self.config.model == self.config.modelA:
            if idx == 0:                    
                exog = self.front_accel if leader_exog == None else leader_exog
            else:
                exog = self.followers[idx-1].x[2] # for model A use the accel of the leading vehicle as exogenous
        
        return exog
    
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
        self.front_accel = util.get_random_val(self.config.rand_gen, self.config.pl_leader_reset_a, std_dev=self.config.pl_leader_reset_a, config=self.config)
       
        for i in range(len(self.followers)):
            self.front_u = util.get_random_val(self.config.rand_gen, self.config.reset_max_u, std_dev=self.config.reset_max_u, config=self.config)

            if i == 0:
                follower_st = self.followers[i].reset(self.front_accel)
            else:
                follower_st = self.followers[i].reset(self.followers[i-1].x[2])

            states.append(follower_st)

        if self.config.framework == self.config.cntrl: 
            states = [list(np.concatenate(states).flat)]

        return states


class Vehicle:
    """Vehicle class based on constant time headway modeling
        Model A and B are functionally the same, just the number of states is only 3 for model A. This means any model implementing this environment should have an input dimension equal to the self.num_states attribute
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
    def __init__(self, idx, config: config.Config, tau_lead=None, a_lead = None, num_states = None, num_actions = None, rand_states=True):
        """constructor - r, h, L and tau referenced from 
                         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7146426/
                         a, b taken from https://www.merl.com/publications/docs/TR2019-142.pdf     
            Arguments:
                idx (integer) : the vehicle's index in the platoon
                config (config.Config) : the configuration class
                tau_lead (float, optional) : the vehicle dynamics coefficent of the leading vehicle in the platoon
                a_lead (float, optional) : the acceleration of the leading vehicle in the platoon 
        """
        log.info(f"=== Inititializing Vehicle {idx}===")
        self.config = config
        self.step_count = 0

        self.idx = idx
        self.h = self.config.timegap
        self.T = self.config.sample_rate
        self.tau = self.config.dyn_coeff
        self.tau_lead = tau_lead
        self.a = self.config.reward_ev_coeff
        self.b = self.config.reward_u_coeff

        self.max_ep = self.config.max_ep # max ep error before environment returns terminate = True
        self.max_ev = self.config.max_ev # max ev error before environment returns terminate = True
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

        self.rand_states = rand_states

        self.num_actions = 1

        self.reset(a_lead) # initializes the states randomly

        """ Defining the system matrices """
        self.set_system_matrices(self.config.method)

    def set_system_matrices(self, method):
        e = np.exp(-self.T/self.tau)
        e_lead = np.exp(-self.T/self.tau_lead)
        if method == self.config.euler:
            print("Using Euler Discretization.")
            self.A = np.array( [ [1, self.T, -self.h*self.T,                                0],
                                 [0,      1, -self.T,                                  self.T],
                                 [0,      0, 1 -(self.T/self.tau),                          0],
                                 [0,      0, 0                   ,  1 -(self.T/self.tau_lead)]])

            self.B = np.array( [0,
                                0,
                                self.T/self.tau,
                                0])
            
            self.C = np.array( [0,
                                0,
                                0,
                                self.T/self.tau_lead])

        elif method == self.config.exact:
            log.info("Using Exact Discretization.")
            A_13 = - self.h * self.tau + self.h * self.tau * e \
            - self.tau * self.T + self.tau**2 - (self.tau ** 2) * e

            A_14 =   self.tau_lead * self.T - self.tau_lead ** 2 \
                    + (self.tau_lead ** 2) * e_lead
            
            A_23 = - self.tau + self.tau * e 

            A_24 =   self.tau_lead - self.tau_lead * e_lead

            self.A = np.array( [ [1, self.T, A_13,   A_14],
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
                               -e_lead + 1])

        log.info(" --- Vehicle %s Initialized System Matrices --- " % (self.idx))
        log.info("A Matrix: %s" % (self.A))
        log.info("B Matrix: %s" % (self.B))
        log.info("C Matrix: %s" % (self.C))
        self.print_hyps(output="log")
        
    def render(self, str_form=False):
        output = f"|x: {np.round(self.x, 2)}, r: {round(self.reward, 2)}, u: {round(self.u, 2)}, exog: {round(self.exog, 2)}|"
        if str_form == True:
            return output
        else:
            print(output, end='\r', flush=True)
    
    def step(self, u, exog_info, debug_mode=False):
        """advances the vehicle model by one timestep

        Args:
            u (float): the action to take
            exog_info (float): the exogenous information given to the vehicle
        """
        self.step_count +=1
        terminal = False
        self.u = u
        self.exog = exog_info 

        if debug_mode:
            print("====__  Vehicle %s __====" % (self.idx))
            self.print_hyps(output="print")
            print("--- Evolution Equation Timestep %s ---" % (self.step_count))
            print("\tA Matrix: ", self.A)
            print("\tx before evolution: ", self.x)

        self.x = self.A.dot(self.x) + self.B.dot(self.u) + self.C.dot(exog_info)

        if abs(self.x[0]) > self.config.max_ep and self.config.can_terminate:
            terminal=True
            self.reward = self.config.terminal_reward  * self.config.re_scalar
        else:
            self.reward = (self.x[0]**2 + self.a*(self.x[1])**2 + self.b*(self.u)**2)  * self.config.re_scalar

        if debug_mode:
            print("\tB Matrix: ", self.B)
            print("\tu: ", self.u)
            print("\tC Matrix: ", self.C)
            print("\texog: ", self.exog)
            print("\tx after evolution: ", self.x)
            print("--- Reward Equation ---")
            print("\tx[0]=epi: ", self.x[0])
            print("\ta: ", self.a)
            print("\tx[1]=evi: ", self.x[1])
            print("\tb: ", self.b)
            print("\tu: ", self.u)


        return self.x[0:self.num_states], -self.reward, terminal # return only the elements that correspond to the state size.
    
    def reset(self, a_lead=None):
        """reset the vehicle environment

        Args:
            a_lead (float, optional): used in models where acceleration is chained throughout the vehicle state. Defaults to None.
        """
        self.u = 0
        self.exog_info = 0
        self.step_count = 0

        if self.rand_states == True:
            self.x = np.array([util.get_random_val(self.config.rand_gen, self.reset_ep_max, std_dev=self.config.reset_ep_max, config=self.config), # intial gap error (m)
                        util.get_random_val(self.config.rand_gen, self.reset_max_ev, std_dev=self.config.reset_max_ev, config=self.config), # initial velocity error
                        util.get_random_val(self.config.rand_gen, self.reset_max_a, std_dev=self.config.reset_max_a, config=self.config), # initial accel of this vehicle
                        a_lead])   # initial accel of leading vehicle
        else:
            self.x = np.array([self.reset_ep_max, # intial gap error (m)
                        self.reset_max_ev, # initial velocity error
                        self.reset_max_a, # initial accel of this vehicle
                        a_lead])   # initial accel of leading vehicle

        return (self.x[0:self.num_states])
    
    def set_state(self, state):
        self.x = state
    
    def print_hyps(self, output: str):
        """
        Method for printing attributes for the class
        """
        valid_output_opts = ["log", "print"]
        if output not in valid_output_opts:
            raise ValueError(f"Invalid parameter for 'output'. Choices are: {valid_output_opts}")

        hyp = " |".join((
                    f"self.idx = {self.idx}",
                    f"self.h = {self.h}",
                    f"self.T = {self.T}",
                    f"self.tau = {self.tau}",
                    f"self.tau_lead = {self.tau_lead}",
                    f"self.a = {self.a}",
                    f"self.b = {self.b}",
                    f"self.max_ep = {self.max_ep}", # max ep error before environment returns terminate = True
                    f"self.max_ev = {self.max_ev}", # max ev error before environment returns terminate = True
                    f"self.reset_ep_max = {self.reset_ep_max}", # used as the max +/- value when generating new ep on reset
                    f"self.reset_max_ev = {self.reset_max_ev}",  # max +/- value when generating new ev on reset
                    f"self.reset_max_a = {self.reset_max_a}" # max +/- value for accel on reset
                ))
        if output == "log":
            log.info(f"---== Hyperparameters for vehicle {self.idx} ==---")
            log.info(hyp)
        elif output == "print":
            print(hyp)

if __name__ == "__main__":
    pass
    # print("hello")
    # conf = config.Config()
    # pl = Platoon(2, conf)
    # # v = Vehicle(1, conf)
    # pl.render()
    # print("\npre reset")
    # print(pl.step([[2.5], [-1.5]]))
    # pl.render()
    # print("\npost")
    # print(pl.reset())

    


    