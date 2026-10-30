"""Rotation features invariant to the engine's arbitrary quaternion sign."""
import numpy as np

def rotation_matrix(quaternion):
    q=np.asarray(quaternion,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all():raise ValueError('Measured finite WXYZ quaternion required')
    length=np.linalg.norm(q)
    if length<1e-12:raise ValueError('A zero quaternion is not an orientation')
    w,x,y,z=q/length
    return np.asarray([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],dtype=np.float32)
