import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_init=None):
        self.theta = theta
        self.mean = mean 
        self.std_dev = std_dev
        self.dt = dt
        self.x_init = x_init
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt 
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x

        return x
    
    def reset(self):
        if self.x_init is not None:
            self.x_prev = self.x_init
        else:
            self.x_prev = np.zeros_like(self.mean)