import numpy as np

def QV_encode(Img_reduced, bitPerPixelGoal):
    #1. calcul des tailles 
    nPix = int(len(Img_reduced) * len(Img_reduced[0]))
    nPixPerVec = 3 #choisis arbitrairement
    bitPerPix = 8 #image a été préprocess pour répondre à ça
    nVec = nPix / nPixPerVec
    closest = np.inf
    closestInd = 0
    for i in range(2,10):
        # la logique du diviseur est en combien de groupe on veux séparer nos vecteur 
        # en assumant que chaque vecteur est différent
        nIndex = (2**i)
        dataSize = nVec*nIndex
        metadataSize = bitPerPix*nPixPerVec*nIndex
        loopBitPerPix = (dataSize+metadataSize)/nPix
        distance = np.abs(bitPerPixelGoal - loopBitPerPix)
        if (distance < closest):
            closest = distance
            closestInd = i

    nIndex = (2**closestInd)
    dataSize = nVec*nIndex
    metadataSize = bitPerPix*nPixPerVec*nIndex
    realBitPerPix = (dataSize+metadataSize)/nPix
    
    #2. 




    I_encoded = Img_reduced
    I_metadata = 0
    return I_encoded, I_metadata, realBitPerPix


#test
if __name__ == "__main__":
    #import for visualisation
    import matplotlib.pyplot as plt
    #generate random data
    imgTest = np.random.randint(0,255,(256,256))

    I_encoded, I_metadata, realEncode = QV_encode(imgTest, 5)

    plt.imshow(imgTest,"Greys")
    plt.show()
