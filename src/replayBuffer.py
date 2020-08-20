from collections import deque
import random 
import numpy as np 

class ReplayBuffer(object):
    """Replay buffer for DDPG learning
    """    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, terminal, next_state):
        """Add experience to the buffer     

        Args:
            state: [description]
            action: [description]
            reward: [description]
            terminal: [description]
            next_state: [description]
        """        
        experience = (state, action, reward, terminal, next_state)

        if self.count < self.size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def size(self):
        return self.count 
    
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        terminal_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, terminal_batch, next_state_batch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0

if __name__=="__main__":
    replayBuffer = ReplayBuffer(1000)
    print(replayBuffer.sample_batch(10))