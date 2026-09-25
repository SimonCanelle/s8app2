from ..QV_encode import QV_encode
from ..QV_decode import QV_decode
from ..computePSNR import computePSNR
import numpy as np
import matplotlib.pyplot as plt

imgTest = np.random.randint(0,255,(256,256))#à changer pour une image source
I_encoded, I_metadata, realBitPerPix = QV_encode(imgTest, 5)
I_decoded = QV_decode(I_encoded, I_metadata)
print("psnr = "+computePSNR(imgTest,I_decoded))

plt.figure("Source")
plt.imshow(imgTest,"Greys")
plt.figure("Decoded")
plt.imshow(I_decoded,"Greys")
plt.show()