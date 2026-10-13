import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    data_type,device=inputs.dtype,inputs.device
    affine=weight_matrix@inputs+biases
    return torch.tanh(affine) if nonlinear else affine
    
