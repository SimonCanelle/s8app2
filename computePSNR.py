import numpy as np


def computePSNR(I_reduced, I_decoded):

    # Difference between reconstructed image and source image
    image_MSE = I_decoded - I_reduced

    # Compute mean squared error
    MSE = np.mean(image_MSE ** 2)

    # PSNR
    if MSE == 0:
        return np.inf

    PSNR = 10 * np.log10(255**2 / MSE)

    return PSNR