import random
import numpy as np

class Platoon:
    def __init__(self, length, T):
        self.length = length
        self.T = T
        self.Vehicles = [Vehicle(i, T) for i in range(0,length)]
        self.idx = 0
        self.front_accel = random.uniform(-3,3)
    
    def __str__(self):
        return str([str(v) for v in self.Vehicles])
    
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
            vehicle = self.Vehicles[i]
            if i < len(self.Vehicles):
                a_lead = self.Vehicles[i+1].x[2]
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
    def __init__(self, idx, T):
        """constructor - r, h, L and tau referenced from 
                         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7146426/
                         a, b taken from https://www.merl.com/publications/docs/TR2019-142.pdf


        Attributes:
            r           (float)    : the constant standstill distance for a vehicle (m)
            h           (float)    : the desired time gap (s)
            L           (float)    : the length of the vehicle (m)
            idx         (int)      : the integer id of the vehicle
            T           (float)    : sample rate
            num_states  (int)      : number of states in the model
            num_actions (int)      : number of actions in the model
            a           (float)    : plant stability constant
            b           (float)    : string stability constant
            x           (np.array) : the state of the system (control error ep, 
                                     control error ev, vehicle acceleration a)
            A           (np.array) : system matrix  
            B           (np.array) : system matrix
            C           (np.array) : system matrix
        """
        self.r = 8
        self.h = 1.25
        self.L = round(random.uniform(2,3), 2)
        self.idx = idx
        self.T = T
        self.tau = 0.2
        self.num_states = 3
        self.num_actions = 1
        self.a = 1
        self.b = 0.1
        self.x = [0]*self.num_states

        self.A = np.array([[1, self.T, -self.T*self.h           ],
                           [0, 1     , -self.T                  ],
                           [0, 0     , 1 - self.T * (1/self.tau)]])
        self.B = np.array([0, 0, self.T*(1/self.tau)])
        self.C = np.array([0, T, 0])
    
    def __str__(self):
        return f"r:{self.r}, h:{self.h}, L:{self.L}"
    
    def step(self, u, a_lead):
        """advances the vehicle model by one timestep

        Args:
            u (float): the action to take
            a_lead (float): the leading vehicles acceleration
        """
        reward = self.x[0]**2 + self.a*(self.x[1])**2 + self.b*(u)**2
        self.x = self.A.dot(self.x) + self.B.dot(u) + self.C.dot(a_lead)
        return self.x, reward
    
    def reset(self):
        self.x = [0]*self.num_states


if __name__=="__main__":
    v = Vehicle(0, 1e-3)
    print(v.x)
    v.step(1.5, 3)
    print(v.x)
    v.step(1.5, 3)
    print(v.x)
    v.step(3, 1)
    print(v.x)
    v.reset()
    print(v.x)
    # platoon = Platoon(5, 1e-3)
    # print(str(platoon))

    


    