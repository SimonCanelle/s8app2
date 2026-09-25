import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from convert import convert
from reduce import reduce
from QV_encode import QV_encode
from QV_decode import QV_decode
from DPCM_encode import DPCM_encode
from DPCM_decode import DPCM_decode
from transmit import transmit
from computePSNR import computePSNR

plt.close("all")


# ==========================================================================
#
# S8 Codage de l'information APP2
#
# La solution est divisée en deux parties :
# - codage
# - décodage
#
# Les données transmises à la couche physique sont regroupées dans Data.
#
# ==========================================================================

# ==========================================================================
#
# SÉLECTION DES PARAMÈTRES
#
# ==========================================================================

# Choix de la quantification
# 1 = Quantification vectorielle (QV)
# 2 = Quantification différentielle (DPCM)
# 3 = Quantification scalaire (QS)
# 4 = Quantification par transformée en cosinus discrète (DCT)
# 5 = Quantification par troncature de blocs (BTC)
# 6 = Quantification adaptative (QA)

Choix = 1

# Chargement de l'image source
I_source = np.asarray(
    # Image.open("ressources/cman.tif"),
    Image.open("ressources/crest.bmp"),
    # Image.open("ressources/irm.tif"),
    # Image.open("ressources/lenna.bmp"),
    # Image.open("ressources/mandrill.tif"),
    dtype=np.float64
)

# Print pour valider que l'image est bien en format 0 à 255 et non 0 à 1
# print(f"min(I_source) = {I_source.min():f}")
# print(f"max(I_source) = {I_source.max():f}")

# Affichage de l'image source
plt.figure(1)
if I_source.ndim == 2:
    # Pour image à 1 canal : Affichage en gris
    plt.imshow(I_source / 255.0, cmap="gray", vmin=0, vmax=1)
else:
    # Pour image à 3 canaux (RGB) : Affichage en couleurs
    plt.imshow(I_source / 255.0)
plt.axis("off")

# Début du chronomètre
start_time = time.perf_counter()


# ==========================================================================
#
# CONVERSION DE FORMAT DE CODAGE DES COULEURS
#
# ==========================================================================

I_source = convert(I_source)


# ==========================================================================
#
# RÉDUCTION DE DIMENSIONS
#
# ==========================================================================

# Dimensions désirées
LIGNES = 256
COLONNES = 256

# Appelle la fonction d'interpolation
I_reduced = reduce(I_source, LIGNES, COLONNES)


# ==========================================================================
#
# CODAGE
#
# ==========================================================================

if Choix == 1:
    # Paramètres d'entrée
    ArgumentX = 0
    # Appelle la fonction de codage
    I_encoded, I_metadata = QV_encode(I_reduced, ArgumentX)

elif Choix == 2:
    # Paramètres d'entrée
    ArgumentX = 0
    # À implémenter
    I_encoded, I_metadata = DPCM_encode(I_reduced, ArgumentX)

elif Choix == 3:
    raise NotImplementedError("Le codeur QS n'est pas encore implémenté.")

elif Choix == 4:
    raise NotImplementedError("Le codeur DCT n'est pas encore implémenté.")

elif Choix == 5:
    raise NotImplementedError("Le codeur BTC n'est pas encore implémenté.")

elif Choix == 6:
    raise NotImplementedError("Le codeur QA n'est pas encore implémenté.")

else:
    raise ValueError("Choix doit être compris entre 1 et 6.")


# ==========================================================================
#
# INTERFACE AVEC LA COUCHE PHYSIQUE
#
# ==========================================================================

# Data représente les données qui seront transmises.
#
# La clé correspond au nombre de bits de la cellule :
#
# Data[8] -> I_encoded
# Data[1] -> I_metadata

Data = {
    8: I_encoded,
    1: I_metadata
}

# Appel de la fonction de transmission
Budget = transmit(Data)

# Si une erreur a été détectée
if Budget < 0:
    print(
        f"Erreur : Une donnée dépasse la gamme dynamique "
        f"à la cellule {-Budget}."
    )


# ==========================================================================
#
# RECOMPOSITION
#
# ==========================================================================

# Dans cette version, Data est conservé directement après la transmission.
# On récupère donc les données reçues à partir de celui-ci.
I_encoded_Rx = Data[8]
I_metadata_Rx = Data[1]


# ==========================================================================
#
# DÉCODAGE
#
# ==========================================================================

if Choix == 1:
    # Paramètres d'entrée
    ArgumentY = 0
    # Appelle la fonction de décodage
    I_decoded = QV_decode(
        I_encoded_Rx,
        I_metadata_Rx,
        ArgumentY
    )

elif Choix == 2:
    # Paramètres d'entrée
    ArgumentY = 0
    # À implémenter
    I_decoded = DPCM_decode(
        I_encoded_Rx,
        I_metadata_Rx,
        ArgumentY
    )

elif Choix == 3:
    raise NotImplementedError("Le décodeur QS n'est pas encore implémenté.")

elif Choix == 4:
    raise NotImplementedError("Le décodeur DCT n'est pas encore implémenté.")

elif Choix == 5:
    raise NotImplementedError("Le décodeur BTC n'est pas encore implémenté.")

elif Choix == 6:
    raise NotImplementedError("Le décodeur QA n'est pas encore implémenté.")


# ==========================================================================
#
# AFFICHAGE DE L'IMAGE DÉCODÉE
#
# ==========================================================================

plt.figure(2)

if I_decoded.ndim == 2:
    plt.imshow(I_decoded / 255.0, cmap="gray", vmin=0, vmax=1)
else:
    plt.imshow(I_decoded / 255.0)

plt.axis("off")


# ==========================================================================
#
# CALCUL DE LA PERFORMANCE
#
# ==========================================================================

# Fin du chronomètre
elapsed_time = time.perf_counter() - start_time

# Si aucune erreur n'a été détectée
if Budget > 0:
    # Calcul du PSNR
    PSNR = computePSNR(I_reduced, I_decoded)
    # Calcul du débit
    Rate = Budget / I_decoded.size
    # Affichage des performances
    print("********* Résultats *********")
    print(f"Temps écoulé: {elapsed_time:.2f} s")
    print(f"PSNR: {PSNR:.2f} dB")
    print(f"Rate: {Rate:.2f} bits/pixel")
    print("*****************************")

# Affiche les figures
plt.show()
