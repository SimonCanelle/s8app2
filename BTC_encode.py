import numpy as np

def BTC_encode(Img_reduced, I_metadata, blocH:int=2, blocL:int=2):
    #1 creation des blocks
    #assume 256x256 en trée
    blockSize = (blocH*blocL)
    blocks = Img_reduced.reshape(-1,blockSize)
    
    #2 calcul des statistiques
    means = np.mean(blocks, axis=1)
    stds = np.std(blocks, axis=1)

    
    #3 encoding
    I_encoded = np.zeros(len(blocks))
    for i, bloc in enumerate(blocks):
        I_encoded[i] = (bloc >= means[i])

    #4 préparation des métadonnées
    I_metadata["means"] = means
    I_metadata["stds"] = stds
    I_metadata["blockSize"] = [blocH, blocL]

    return I_encoded, I_metadata

