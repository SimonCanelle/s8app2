import numpy as np


def distanceEuclidienne(a,b):
    diff = a - b
    c = np.sum(diff**2)
    return np.sqrt(c)

def closestVector(imgVec, encodingTable):
    diff = encodingTable - imgVec          
    dist_sq = np.sum(diff**2, axis=1)
    return np.argmin(dist_sq)

def QV_encode(Img_reduced, I_metadata, pixelPerVec:int=10, bitPerIndex:int=10, conversionGoal=0.01):
    #1. vérification de la taille de l'image
    imgH = len(Img_reduced)
    imgL = len(Img_reduced[0])    

    #2. initialisation du tableau d'encodage
    nIndex = int(2**bitPerIndex)    
    
    #3. vectorisation de l'image
    # j'ai décider de séparer l'image en vecteurs colonne pour simplifier 
    # la forme pour des vecteurs de grandeurs différentes
    heightCheck = imgH % pixelPerVec #vérification de size de tableau
    if (heightCheck!=0):
        for i in range(0,(pixelPerVec-heightCheck)):
            Img_reduced = np.vstack([Img_reduced, Img_reduced[-1:]])#copie la dernière ligne 
    
    imgVec = np.zeros((int(Img_reduced.size/pixelPerVec),pixelPerVec))
    i = 0
    for j in range(0,imgH,pixelPerVec):
        for k in range(0,imgL):
            imgVec[i] = Img_reduced[j:j+pixelPerVec, k]
            i+=1            

    uniqueVec = np.unique(imgVec, axis=0)
    if (len(uniqueVec)>=nIndex) : 
        uniqueIndexes = np.random.choice(len(uniqueVec), nIndex, replace=False)
        encTab = uniqueVec[uniqueIndexes].copy()
    else:
        encTab = np.zeros((nIndex,pixelPerVec))
        line = np.linspace(Img_reduced.min(),Img_reduced.max(),nIndex) #de min a max trouvé dans l'image
        counter = 0 #aide a pas avoir des vecteur pareil au départ
        for i in range(0,nIndex):
            n = int(np.round(line[i]))
            encTab[i] = np.full(pixelPerVec,n)
            for j in range(1,pixelPerVec):
                encTab[i][j] = (n + counter)%256
                counter = (counter + 1) % 16

    #boucle LBG commence ici, pourra être changé
    for lbg in range(0, 25):
        #4. encodage 
        I_encoded = np.zeros(len(imgVec))
        for i in range(0,len(I_encoded)):
            I_encoded[i] = closestVector(imgVec[i], encTab)

        #5. recalcul des centroïdes
        encTabOld = encTab.copy() #backup pour calcul de convergence
        for i in range(0,2**bitPerIndex):
            indexes = np.where(I_encoded == i)[0]
            if len(indexes) != 0:
                moy = np.round(np.mean(imgVec[indexes], axis=0))
                encTab[i] = moy

        #6. calcul de convergence
        convTab = np.zeros(len(encTab))
        for i in range(0,len(encTab)):
           dist = distanceEuclidienne(encTabOld[i], encTab[i]) #calcul le mouvement
           convTab[i] = np.linalg.norm(dist) #calcul la norme du vecteur de mouvement
        conv = np.mean(convTab)
        if conv < conversionGoal:#set arbitrairement
           break #la convergence est terminé

        #7. vérification de classe vide
        if len(np.unique(I_encoded)) < len(encTab):
           indTab = np.arange(0,2**bitPerIndex,1) #liste des valeurs de 0 à 2^nBitPerInd
           unusedInd = np.setdiff1d(I_encoded,indTab) #liste des indexes non utilisé        
           #take most used class
           #move vector -std and +std
           #one goes in the most used class
           #other goes in unused class
           histo, bins = np.histogram(I_encoded, bins=2**bitPerIndex)
           for i in unusedInd:
               maxIndex = bins[np.argmax(histo)]
               maxVec = np.where(I_encoded == maxIndex)[0]
               maxStd = np.std(maxVec, axis=0)
               maxMean = np.mean(maxVec, axis=0)
               encTab[maxIndex] = maxMean - maxStd #modify max class
               encTab[i] = maxMean + maxStd #modify unused class
               histo[maxIndex] = 0 #remove most used histogram from list
    
    I_encoded = np.zeros(len(imgVec))
    for i in range(0,len(I_encoded)):
        I_encoded[i] = closestVector(imgVec[i], encTab)
    
    I_metadata["encodingTable"] = encTab

    return I_encoded, I_metadata

    
