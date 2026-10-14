import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    
    frequencies=(1/base)**((2/d_model)*np.arange((d_model+1)//2))[None,:]
    position=np.arange(seq_len)[...,None]
    pe=np.zeros((int(seq_len),d_model),dtype=float)
    mat=position@frequencies
    mat_sin=np.sin(mat)
    mat_cos=np.cos(mat)
    pe[:,0::2]=mat_sin
    if d_model%2!=0:
        pe[:,1::2]=mat_cos[:,:-1]
    else:
        pe[:,1::2]=mat_cos
    
    return pe
    
    
    
    
        
                           
 