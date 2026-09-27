from convert import convert
from reduce import reduce
from QV_encode import QV_encode
from QV_decode import QV_decode
from transmit import transmit
from computePSNR import computePSNR
import numpy as np
from PIL import Image

#algorithme de recherche des meilleurs paramètres pour optimiser le psnr

imgH = 256
imgL = 256
bitPerPixelGoal = 5
nPix = int(imgH * imgL)
bitPerPix = 8 #image a été préprocess pour répondre à ça
params_pairs = None#PixPerVec, BitPerInd, loopBitPerPix
for PixPerVec in range(2,12):
    imgHPadded = imgH + (imgH % PixPerVec)
    nPixPadded = imgHPadded * imgL
    nVec = nPixPadded / PixPerVec
    for BitPerInd in range(2,12):
        dataSize = nVec*BitPerInd
        metadataSize = bitPerPix*PixPerVec*2**BitPerInd + 96 #header: taille image source[16,16] + taille tableau encodage[16,16] + taille image encodé [16,16]
        # la logique du diviseur est en combien de groupe on veux séparer nos vecteur 
        # en assumant que chaque vecteur est différent            
        loopBitPerPix = (dataSize+metadataSize)/nPix
        if (loopBitPerPix <= 5 and loopBitPerPix >= 2):# pas moins que 2 car c'Est sur que la uqalité va être ouach
            print(f"Ok Pairs : P/V {PixPerVec} B/I {BitPerInd}")
            print(f"calculated data bit size = {dataSize+metadataSize}")
            print(f"pre calculated rate = {loopBitPerPix}")
            info = [PixPerVec, BitPerInd, loopBitPerPix]
            if params_pairs is None:
                params_pairs = info
            else:
                params_pairs = np.vstack([params_pairs,info])


I_source = np.asarray(
    # Image.open("ressources/cman.tif"),
    #Image.open("ressources/crest.bmp"),
    # Image.open("ressources/irm.tif"),
    Image.open("ressources/lenna.bmp"),
    # Image.open("ressources/mandrill.tif"),
    dtype=np.float64
)
I_source = convert(I_source)
LIGNES = 256
COLONNES = 256
# Appelle la fonction d'interpolation
I_reduced = reduce(I_source, LIGNES, COLONNES)


bestPSNR = 0
bestParam = []
for param in params_pairs:
    print(f"Testing {param}")
    I_metadata = dict()
    I_metadata["src_image_size"] = [I_source.shape[0],I_source.shape[1]]
    I_metadata["enc_image_size"] = [LIGNES,COLONNES]
    I_encoded, I_metadata = QV_encode(I_reduced, I_metadata,int(param[0]),int(param[1]))
    #préparation des données comme dans le main
    Data = [None] * 16
    #metadonnées
    # 6x16 bit header + MxNx8 bit enc table 
    Data[15] = I_metadata["src_image_size"].copy() #[image height, image width]
    Data[15] += I_metadata["enc_image_size"] #[image height, image width]
    Data[15] += [I_metadata["encodingTable"].shape[0],I_metadata["encodingTable"].shape[1]]#[table height, table width]
    Data[7] = I_metadata["encodingTable"].flatten()
    #données
    #data indexes dépends de la taille du tableau
    index = int(np.log2(I_metadata["encodingTable"].shape[0]))-1
    if Data[index] is not None and Data[index].size > 0 :
        Data[index] = np.concatenate([Data[index],I_encoded])
    else:
        Data[index] = I_encoded

    I_decoded, I_metadata_dec = QV_decode(Data)
    psnr = computePSNR(I_reduced,I_decoded)
    print(f"result psnr : {psnr}")

    if psnr > bestPSNR:
        bestPSNR = psnr
        bestParam = param

print(f"Best PSNR {bestPSNR}, P/V {bestParam[0]}, B/I {bestParam[1]}, bitRate {bestParam[2]}")