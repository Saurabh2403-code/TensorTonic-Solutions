import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    points=np.asarray(points)
    vector=(len(points.shape)==1)
    if vector==1:
        points=points.reshape(1,3)

    ones=np.ones((points.shape[0],1))
    points=np.concatenate([points,ones],axis=1)
    T=np.asarray(T)
    out=(T@points.T).T[:,:3]
    return out[0] if vector else out