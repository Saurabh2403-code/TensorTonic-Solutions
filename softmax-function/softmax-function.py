import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x=np.array(x)
    
    x=x-(np.max(x,axis=-1)[...,None])
    exp_x=np.exp(x)
    sum=np.sum(exp_x,axis=-1)[...,None]
    return exp_x/sum