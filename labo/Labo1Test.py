import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from interp_bilineaire import interp_bilineaire


# ============================================================
# P#1 : Interpolation bilinéaire
# ============================================================

# Select image
img = 5

if img == 1:
    I_read = np.array(Image.open("ressources/cman.tif"))
elif img == 2:
    I_read = np.array(Image.open("ressources/mandrill.tif"))
elif img == 3:
    I_read = np.array(Image.open("ressources/irm.tif"))
elif img == 4:
    I_read = np.array(Image.open("ressources/lenna.bmp"))
else:
    I_read = np.array(Image.open("ressources/crest.bmp"))


# Convert RGB image to grayscale if necessary
if I_read.ndim == 3:
    I_source = np.array(Image.fromarray(I_read).convert("L"))
else:
    I_source = I_read

# Make sure the image is uint8
I_source = I_source.astype(np.uint8)

# Display minimum and maximum
print(f"min(I_source) = {I_source.min():f}")
print(f"max(I_source) = {I_source.max():f}")


# ============================================================
# Display original images
# ============================================================

plt.figure(1)
plt.imshow(I_read, cmap="gray" if I_read.ndim == 2 else None)
plt.axis("off")
plt.show(block=False)

plt.figure(2)
plt.imshow(I_source, cmap="gray")
plt.axis("off")
plt.show(block=False)


# ============================================================
# Get original image dimensions
# ============================================================

L1, C1 = I_source.shape

print(f"L source = {L1:f}")
print(f"C source = {C1:f}")


# ============================================================
# Compress
# ============================================================

L2 = 128
C2 = L2

I_redim_1 = interp_bilineaire(I_source, L2, C2)

L_redim_1, C_redim_1 = I_redim_1.shape
print(f"L shrink = {L_redim_1:f}")
print(f"C shrink = {C_redim_1:f}")

plt.figure(3)
plt.imshow(I_redim_1, cmap="gray")
plt.axis("off")
plt.show(block=False)


# ============================================================
# Expand
# ============================================================

L3 = L1
C3 = C1

I_redim_2 = interp_bilineaire(I_redim_1, L3, C3)

L_redim_2, C_redim_2 = I_redim_2.shape
print(f"L expand = {L_redim_2:f}")
print(f"C expand = {C_redim_2:f}")


plt.figure(4)
plt.imshow(I_redim_2, cmap="gray")
plt.axis("off")
plt.show()

