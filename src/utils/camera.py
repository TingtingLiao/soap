import numpy as np 

 
class Camera:
    def __init__(self, phi=0, theta=0): 
        self.phi = phi
        self.theta = theta
        self.s = 1
 
    @property
    def extrinsic(self): 
        M = np.eye(4, dtype=np.float32)
        R = make_rotate(np.radians(self.theta), 0, 0) @ make_rotate(0, np.radians(self.phi), 0)
        M[:3, :3] = R
        return M
    
    @property
    def intrinsics(self):
        return np.array([
            [self.s, 0, 0, 0],
            [0, self.s, 0, 0],
            [0, 0, self.s, 0],
            [0, 0, 0, 1]], dtype=np.float32
            ) 
        
    @property
    def mvp(self):
        return self.intrinsics @  self.extrinsic  # [4, 4]

    def orbit(self, dx, dy): 
        self.phi += np.radians(dx)
        self.theta += np.radians(dy)
        
    def scale(self, delta):
        self.s *= 1.1 ** delta

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])