import numpy as np
from .basic import Traj_Base

class Rot(Traj_Base):
    
    def camera_target_up(self):
        t = np.linspace(0, 1, self.nframe)
        # rotation angles at each frame
        theta = 2 * np.pi * t * self.nframe 
        # try not to change y (up-down for floor and sky)
        pos = np.zeros((self.nframe,3))
        x = np.sin(theta)
        y = np.zeros_like(x)
        z = np.cos(theta)
        targets = np.vstack([x,y,z]).T
        camera_ups = np.array([[0,-1.,0]]).repeat(self.nframe,axis=0)
        # camera_triples
        cameras = []
        for i in range(self.nframe):
            cameras.append((pos[i],targets[i],camera_ups[i]))

        for i in range(len(cameras)):
            C, T, U = cameras[i]
            C[0] = 0.0
            C[1] = 0.0
            C[2] = 0.0
            cameras[i] = (C, T, U)

        return cameras
    