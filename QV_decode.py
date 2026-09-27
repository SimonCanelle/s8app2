import numpy as np
def QV_decode(Data):
    #décodage des métadonnées
    I_metadata = dict()
    I_metadata["src_image_size"] = Data[15][0:2] #[image height, image width]
    I_metadata["enc_image_size"] = Data[15][2:4]#[image height, image width]
    encTableShape = Data[15][4:6]
    I_metadata["encodingTable"] = np.asarray(Data[7][0:int(encTableShape[0]*encTableShape[1])]).reshape(encTableShape[0],encTableShape[1]) 

    #décodage des données
    nBitInIndex = int(np.log2(encTableShape[0]))
    nPixelPerVec = encTableShape[1]
    if nBitInIndex == 8:
        I_encoded = np.asarray(Data[7][int(encTableShape[0]*encTableShape[1]):-1])
    elif nBitInIndex == 16:
        I_encoded = np.asarray(Data[15][6:-1])
    else:
        I_encoded = np.asarray(Data[nBitInIndex-1])

    #lecture des métadonnées
    enTab = I_metadata["encodingTable"]
    imgH, imgL = I_metadata["enc_image_size"]       
    #check du nombre de pixel par vecteur
    nPixPerVec = len(enTab[0])
    # j'ai décider de séparer l'image en vecteurs colonne pour simplifier 
    # la forme pour des vecteurs de grandeurs différentes
    heightCheck = imgH % nPixPerVec #vérification de size de tableau
    #préparation de l'array pour l'image
    I_decoded = np.zeros((imgH+(nPixPerVec-heightCheck),imgL)) 
    x = 0   #ligne 
    y = 0   #colonne
    #reconstruction de l'image
    for i in range(0,len(I_encoded)):
        I_decoded[x:x+nPixPerVec,y] = enTab[int(I_encoded[i])]
        y=i%imgL
        x=int(np.floor(i/imgL)*nPixPerVec)        
    #retire les ligne non voulue
    I_decoded = I_decoded[0:imgH,:]
    return I_decoded, I_metadata