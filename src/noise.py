import numpy as np
from src import util
class OUActionNoise:
    def __init__(self, mean, x_init=None, config=None):
        self.config = config
        self.theta = self.config.theta
        self.mean = mean 
        
        self.std_dev = float(self.config.std_dev)* np.ones(1)
        self.dt = self.config.ou_dt
        self.x_init = x_init
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt 
            + self.std_dev * np.sqrt(self.dt) * util.get_random_val(self.config.normal, std_dev=1.0, config=self.config, size=self.mean.shape)
        )

        self.x_prev = x

        return x
    
    def reset(self):
        if self.x_init is not None:
            self.x_prev = self.x_init
        else:
            self.x_prev = np.zeros_like(self.mean)