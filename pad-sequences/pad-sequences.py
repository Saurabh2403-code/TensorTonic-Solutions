import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs)==0:
        return np.empty((0,0),int)
    if max_len:
        max_len=max_len
    else:
        max_len=max(len(item) for item in seqs)
    final_seq=[]
    for i in range(len(seqs)):
        if len(seqs[i])==max_len:
            final_seq.append(seqs[i])
            print(f'equal:{final_seq}')
        elif len(seqs[i])<max_len:
            for j in range(len(seqs[i]),max_len):
                seqs[i].append(pad_value)
            final_seq.append(seqs[i])
            print(f'less:{final_seq}')
        else:
            final_seq.append(seqs[i][:max_len])
            print(f'greater:{final_seq}')
    print(final_seq)
    return np.asarray(final_seq,int)
            
        
    
            
        
    