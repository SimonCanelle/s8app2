import numpy as np
from PIL import Image

def convert(I_source):
    # If 3 canals, convert to grayscale
    if (I_source.ndim == 3):
        I_source = I_source.astype(np.uint8)
        I_source = np.array(Image.fromarray(I_source).convert("L"))
    # Cast as uint8
    I_source = I_source.astype(np.uint8)
    
    return I_source