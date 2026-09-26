import numpy as np
import matplotlib.pyplot as plt

def DPCM_encode(I_reduced, ArgumentX):
    # delta = 1.0873 # Dict size = 4, Laplacian distribution, 2^2
    delta = 0.4609 # Dict size = 16, Laplacian distribution 2^4
    # delta = 0.3352 # Dict size = 16, Gaussian distribution 2^4
    # delta = 0.217 # Dict size = 16, Uniform distribution 2^4

    L, C = I_reduced.shape
    I_encoded = np.zeros((L, C), dtype=np.uint8)
    I_reconstructed = np.zeros((L, C), dtype=float)

    mean, std = DPCM_mean_and_std(I_reduced)
    # print("mean =", mean)
    # print("std =", std)

    MEAN_MIN = -0.5
    MEAN_MAX = 0.5
    STD_MIN = 0
    STD_MAX = 64

    mean_code = np.uint8(quantize_value(mean, MEAN_MIN, MEAN_MAX, 8))
    std_code = np.uint8(quantize_value(std, STD_MIN, STD_MAX, 8))
    mean_reconstructed = dequantize_value(mean_code, MEAN_MIN, MEAN_MAX, 8)
    std_reconstructed = dequantize_value(std_code, STD_MIN, STD_MAX, 8)

    I_metadata = (mean_code, std_code)

    for l in range(L):
        for c in range(C):

            # Prediction
            prediction = DPCM_predictor(I_reconstructed, l, c)

            # Prediction error
            erreur = float(I_reduced[l, c]) - prediction

            # Normalization of the error
            erreur_normalisee = (erreur - mean_reconstructed) / std_reconstructed

            # Quantization
            code = DPCM_quantificator(erreur_normalisee, delta)

            # Write the encoded data
            I_encoded[l, c] = code

            # Dequantization
            erreur_normalisee_reconstruite  = DPCM_dequantificator(code, delta)

            # Unormalization of the error
            erreur_reconstruite = erreur_normalisee_reconstruite * std_reconstructed + mean_reconstructed

            # Reconstruction
            I_reconstructed[l, c] = np.clip(prediction + erreur_reconstruite, 0, 255)

    return I_encoded, I_metadata

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

def DPCM_mean_and_std(I_reduced):
    L, C = I_reduced.shape
    I_reconstructed = np.zeros((L, C), dtype=float)
    I_erreur = np.zeros((L, C), dtype=float)

    for l in range(L):
        for c in range(C):
            # Prediction
            prediction = DPCM_predictor(I_reconstructed, l, c)
            # Prediction error
            I_erreur[l, c] = float(I_reduced[l,c]) - prediction
            I_reconstructed[l, c] = I_reduced[l,c]

    # plt.figure(4)
    # plt.hist(I_erreur.flatten(), bins=200, density=True)
    # plt.xlabel("Erreur de prédiction")
    # plt.ylabel("Densité")
    # plt.title("Distribution des erreurs DPCM")

    mean = np.average(I_erreur)
    std = np.std(I_erreur)

    return mean, std

#############################################################################################

def DPCM_quantificator(val_ini, delta):

    seuils_decision = np.array([
        -np.inf,
        -7 * delta,
        -6 * delta,
        -5 * delta,
        -4 * delta,
        -3 * delta,
        -2 * delta,
        -delta,
        0,
        delta,
        2 * delta,
        3 * delta,
        4 * delta,
        5 * delta,
        6 * delta,
        7 * delta,
        np.inf
    ])

    codes = np.arange(16)

    indice = np.digitize(
        val_ini,
        seuils_decision[1:-1],
        right=False
    )

    val_quant = codes[indice]

    return val_quant

#############################################################################################

def DPCM_dequantificator(code, delta):

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

    val_reconstruite = niveaux_reconstruction[code]

    return val_reconstruite

#############################################################################################

def quantize_value(value, minimum, maximum, n_bits):
    n_levels = 2**n_bits
    delta = (maximum - minimum) / n_levels

    code = int(np.floor((value - minimum) / delta))
    code = np.clip(code, 0, n_levels - 1)

    return code

#############################################################################################

def dequantize_value(code, minimum, maximum, n_bits):
    n_levels = 2**n_bits
    delta = (maximum - minimum) / n_levels

    value = minimum + (code + 0.5) * delta

    return value