import numpy as np

def QV_encode(Img_reduced, bitPerPixelGoal):
    #1. calcul de la taille optimale du tableau pour le nombre de bit par pixel voulu
    imgH = len(Img_reduced)
    imgL = len(Img_reduced[0])
    nPix = int(imgH * imgL)
    bitPerPix = 8 #image a été préprocess pour répondre à ça
    nPixPerVec = 0
    nBitPerInd = 0
    realBitPerPix = 0
    closest = np.inf
    for j in range(2,8):
        nVec = nPix / j        
        for i in range(2,10):            
            nIndex = (2**i)
            dataSize = nVec*i
            metadataSize = bitPerPix*nPixPerVec*nIndex + 64 #header: taille image source[16,16] + taille tableau encodage[16,16]
            # la logique du diviseur est en combien de groupe on veux séparer nos vecteur 
            # en assumant que chaque vecteur est différent
            loopBitPerPix = (dataSize+metadataSize)/nPix
            distance = np.abs(bitPerPixelGoal - loopBitPerPix)
            if (distance < closest):
                closest = distance
                nBitPerInd = i
                nPixPerVec = j
                realBitPerPix = loopBitPerPix

    #2. initialisation du tableau d'encodage de manière linéaire
    nIndex = 2**nBitPerInd
    encTab = np.zeros((nIndex,nPixPerVec))
    line = np.linspace(0,255,nIndex)
    for i in range(0,nIndex):
        n = int(np.round(line[i]))
        encTab[i] = np.full(nPixPerVec,n)
    
    #3. vectorisation de l'image
    # j'ai décider de séparer l'image en vecteurs colonne pour simplifier 
    # la forme pour des vecteurs de grandeurs différentes
    imgVec = np.zeros((int(nPix/nPixPerVec),nPixPerVec))
    i = 0
    for j in range(0,imgH,nPixPerVec):
        for k in range(0,imgL):
            imgVec[i] = Img_reduced[j:j+nPixPerVec, k]
            i+=1

    I_encoded = imgVec
    I_metadata = encTab
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
