import numpy as np

from config import *


class KalmanObjectTracker(object):
    count = 1
    def __init__(self, bbox):
        r"""
            state: [u, v, s, r, u_dot, v_dot, s_dot]
            we assume the aspect ratio of any object to be constant.
        """
        self.bbox = bbox
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = bbox_to_z(bbox)
        self.F = np.array([
            [1., 0., 0., 0., 1., 0., 0.], 
            [0., 1., 0., 0., 0., 1., 0.], 
            [0., 0., 1., 0., 0., 0., 1.], 
            [0., 0., 0., 1., 0., 0., 0.], 
            [0., 0., 0., 0., 1., 0., 0.], 
            [0., 0., 0., 0., 0., 1., 0.], 
            [0., 0., 0., 0., 0., 0., 1.]
        ], dtype=np.float32)
        self.P = np.eye(7, dtype=np.float32)
        self.P *= 10.
        self.P[4:, 4:] *= 1000.
        self.P[-1, -1] = .00000001  # To keep scale velocity zero
        self.R = np.eye(7, dtype=np.float32)
        self.R[2:, 2:] *= 2.
        self.R[-1, -1] = .00000001  # To keep scale velocity zero
        self.Q = np.eye(7, dtype=np.float32)
        self.Q *= .01
        self.age = 0
        self.maturity = 0
        self.id = None
        self.detected = False  # To be used for displaying only detected ones
    
    def predict(self):
        if self.id is None:
            self.id = KalmanObjectTracker.count
            KalmanObjectTracker.count += 1
        self.age += 1
        self.x = self.F @ self.x
        self.P = (self.F @ self.P @ self.F.T) + self.Q
        return z_to_bbox(self.x[:4]), self.P
    
    def update(self, bbox):
        self.age = 0
        self.maturity += 1
        z_k = bbox_to_z(bbox)
        z_k_1 = bbox_to_z(self.bbox)
        z = np.zeros((7, 1), dtype=np.float32)
        z[:4] = z_k
        z[4:6] = z_k[:2] - z_k_1[:2]  # Keep scale velocity constant
        # z[4:7] = z_k[:3] - z_k_1[:3]  # Comment to keep scale velocity constant
        K = self.P @ np.linalg.inv((self.P + self.R))
        self.x = self.x + K @ (z - self.x)
        self.P = self.P - (K @ self.P)
        self.bbox = bbox
    
    def get_state(self):
        return z_to_bbox(self.x[:4])
    
    def is_mature(self):
        return True if self.maturity > maturity_period else False
    
    def expired(self):
        return True if self.age > max_age else False


def bbox_to_z(bbox):
    r"""
        bbox: [x1, y1, x2, y2]
        returns: [u, v, s, r]
    """
    bbox = bbox.astype(np.float32)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w/2
    v = bbox[1] + h/2
    s = h * w
    r = w / h
    return np.array([[u], [v], [s], [r]], dtype=np.float32)

def z_to_bbox(z):
    r"""
        z: [u, v, s, r]
        returns: [x1, y1, x2, y2]
    """
    z = z.reshape(4)
    w= np.sqrt(z[2] * z[3])
    h = z[2]/w
    x1 = z[0] - w/2
    y1 = z[1] - h/2
    x2 = z[0] + w/2
    y2 = z[1] + h/2
    return np.array([x1, y1, x2, y2], dtype=np.float32)
