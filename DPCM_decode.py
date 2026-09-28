import numpy as np

def DPCM_decode(I_encoded, I_metadata, ArgumentY):
    # delta = 1.0873 # Dict size = 4, Laplacian distribution, 2^2
    delta = 0.4609 # Dict size = 16, Laplacian distribution 2^4
    # delta = 0.3352 # Dict size = 16, Gaussian distribution 2^4
    # delta = 0.217 # Dict size = 16, Uniform distribution 2^4

    L, C = I_encoded.shape

    I_decoded = np.zeros((L, C), dtype=float)
    I_decoded_8bits = np.zeros((L, C), dtype=np.uint8)

    # Metadata
    mean_code, std_code = I_metadata

    # Limits to the quantification of mean and standard deviation
    MEAN_MIN = -0.5
    MEAN_MAX = 0.5
    STD_MIN = 0
    STD_MAX = 64

    mean = dequantize_value(mean_code, MEAN_MIN, MEAN_MAX, 8)
    std = dequantize_value(std_code, STD_MIN, STD_MAX, 8)

    for l in range(L):
        for c in range(C):

            # Prediction
            prediction = DPCM_predictor(I_decoded, l, c)

            # Dequantization
            erreur_normalisee_reconstruite = DPCM_dequantificator(I_encoded[l, c], delta)

            # Unnormalize the error
            erreur_reconstruite = (erreur_normalisee_reconstruite * std + mean)

            # Reconstruction
            I_decoded[l, c] = np.clip(prediction + erreur_reconstruite, 0, 255)
            I_decoded_8bits[l, c] = int(round(I_decoded[l, c]))

    return I_decoded_8bits

#############################################################################################

def DPCM_predictor(I_reconstructed, l, c):
    # First pixel
    if l == 0 and c == 0:
        prediction = 128

    # First row
    elif l == 0:
        prediction = I_reconstructed[l, c - 1]

    # First column
    elif c == 0:
        prediction = I_reconstructed[l - 1, c]

    # Interior pixels: median predictor
    else:
        p1 = (0.5 * I_reconstructed[l - 1, c] + 0.5 * I_reconstructed[l, c - 1])
        p2 = (0.5 * I_reconstructed[l - 1, c - 1] + 0.5 * I_reconstructed[l, c - 1])
        p3 = (0.5 * I_reconstructed[l - 1, c - 1] + 0.5 * I_reconstructed[l - 1, c])
        prediction = np.median([p1, p2, p3])

    return prediction

#############################################################################################

def DPCM_dequantificator(code, delta):

    # Static table for reconstruction levels of 4 bits (2^4 = 16 levels)
    niveaux_reconstruction = np.array([
        -15 * delta / 2,
        -13 * delta / 2,
        -11 * delta / 2,
        -9 * delta / 2,
        -7 * delta / 2,
        -5 * delta / 2,
        -3 * delta / 2,
        -delta / 2,
         delta / 2,
         3 * delta / 2,
         5 * delta / 2,
         7 * delta / 2,
         9 * delta / 2,
         11 * delta / 2,
         13 * delta / 2,
         15 * delta / 2
    ])

    # Finds correct value in code based on index
    val_reconstruite = niveaux_reconstruction[code]

    return val_reconstruite

#############################################################################################

def dequantize_value(code, minimum, maximum, n_bits):
    # QSU-1 for the mean and standard deviation
    n_levels = 2**n_bits
    delta = (maximum - minimum) / n_levels

    value = minimum + (code + 0.5) * delta

    return value