import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    shape=np.shape(X)#(n_samples,n_features)
    weights=np.zeros(shape[-1],)#(n_features,)
    bias=np.array(0.0)
    for i in range(steps):
        z=X@weights+bias[None]#(n_samples,)
        pred=_sigmoid(z)
        error=(pred-y)
        dl_dw=(X.T@error)/(len(error))
        dl_db=np.sum(error)/(len(error))
        weights=weights-(lr*dl_dw)
        bias=bias-(lr*dl_db)
    return (weights,bias)
        
        
        