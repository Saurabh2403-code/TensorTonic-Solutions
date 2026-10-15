import numpy as np
#ok i got the probabilities for the individual element for the input but how do I drop the element following that probability distribution? I create a mask making the element to be dropped zero and the rest 1/(1-p) but still the core problem is how do I sample values from a distribution
#
def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x=np.asarray(x,float)
    rng=rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0)

    keeping_probability=1-p
    mask=(rng.random(x.shape)<keeping_probability).astype(float)/keeping_probability
    return x*mask,mask