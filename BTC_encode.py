import numpy as np

def BTC_encode(Img_reduced, I_metadata):
    blocH = 2
    blocL = 2
    #1 creation des blocks
    #assume 256x256 en trée
    blockSize = (blocH*blocL)
    nBlocks = int(Img_reduced.size / blockSize)
    blocks = [None]*nBlocks
    i=0
    for x in range(0,256,2):
         for y in range(0,256,2):
            blocks[i] = Img_reduced[x:x+2,y:y+2].flatten()
            i+=1
    
    #2 calcul des statistiques
    means = np.mean(blocks, axis=1)
    stds = np.std(blocks, axis=1)

    
    #3 encoding
    I_encoded = np.zeros((len(blocks),4))
    for i, bloc in enumerate(blocks):
        I_encoded[i] = (bloc >= means[i])

    #4 préparation des métadonnées
    I_metadata["means"] = means
    I_metadata["stds"] = stds

    return I_encoded, I_metadata

