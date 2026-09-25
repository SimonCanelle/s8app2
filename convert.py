import numpy as np
from PIL import Image

def convert(I_source):

    if (I_source.ndim == 3):
        I_source = I_source.astype(np.uint8)
        I_source = np.array(Image.fromarray(I_source).convert("L"))
        
    I_source = I_source.astype(np.uint8)
    
    return I_source