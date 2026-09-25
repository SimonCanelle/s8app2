from QV_encode import QV_encode
from QV_decode import QV_decode
from computePSNR import computePSNR
import numpy as np
import matplotlib.pyplot as plt
import cv2

#imgTest = np.random.randint(0,255,(256,256))#à changer pour une image source
imgSource = cv2.imread('ressources/lenna.bmp', cv2.IMREAD_GRAYSCALE)  # (H, W)
imgTest = cv2.resize(imgSource, (256, 256))
I_encoded, I_metadata, realBitPerPix = QV_encode(imgTest, 5, 8, 12)
I_decoded = QV_decode(I_encoded, I_metadata)
print("bit/pixel = "+ str(realBitPerPix))
print("psnr = "+str(computePSNR(imgTest,I_decoded)))

plt.figure("Source")
plt.imshow(imgTest,"gray", vmin=0, vmax=255 )
plt.figure("Decoded")
plt.imshow(I_decoded,"gray", vmin=0, vmax=255)
plt.figure("bruit")
plt.imshow(imgTest - I_decoded,"gray", vmin=0, vmax=255)
plt.show()