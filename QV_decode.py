import numpy as np
def QV_decode(I_encoded, I_metadata):
    #lecture des métadonnées
    enTab = I_metadata["encodingTable"]
    imgH, imgL = I_metadata["image_size"]       
    #check du nombre de pixel par vecteur
    nPixPerVec = len(enTab[0])
    # j'ai décider de séparer l'image en vecteurs colonne pour simplifier 
    # la forme pour des vecteurs de grandeurs différentes
    heightCheck = imgH % nPixPerVec #vérification de size de tableau
    #préparation de l'array pour l'image
    I_decoded = np.zeros(imgH+heightCheck,imgL) 
    x = 0   #ligne 
    y = 0   #colonne
    #reconstruction de l'image
    for i in range(0,len(I_encoded)):
        I_decoded[x:x+nPixPerVec,y] = enTab[i]
        y+=1
        x+=nPixPerVec        
    #retire les ligne non voulue
    I_decoded = I_decoded[0:imgH,:]
    return I_decoded