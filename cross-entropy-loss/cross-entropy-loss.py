import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_true=np.asarray(y_true,int)
    y_pred=np.asarray(y_pred,float)
    pred_selected=y_pred[np.arange(len(y_true)),y_true]
    log_prob=-np.log(pred_selected)
    sum=np.sum(log_prob)
    return sum/len(y_true)