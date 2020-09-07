class Platoon:

    def __init__(self, length, dt):
        self.length = length
        self.dt = dt
    

class Vehicle:
    """Vehicle class based on constant time headway modeling
    """
    def __init__(self, r, h, L):
        """constructor

        Args:
            r    (float): the constant standstill distance for a vehicle
            tgap (float): the desired time gap
            L    (float): the length of the vehicle
        """
        self.r = r
        self.h = h
        self.L = L
        self.p_curr = 0
        self.p_last = 0

    def get_vel(self):
        pass

    def get_headway(self, p_lead, p_self):
        d = p_lead - p_self - self.L 
        return d

    def get_desired_headway(self, p_lead, p_self):
        dr = self.r + self.h * self.get_vel()
        return dr

    


    