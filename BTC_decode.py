import numpy as np

def BTC_decode(Data):
    blocH = 2
    blocL = 2
    blockSize = (blocH*blocL)
    #I_metadata_dec = Data[7]
    I_encoded = Data[3].astype(np.uint32)

    nBlocks = len(I_encoded)

    pixelPerBloc = blocH*blocL

    means = Data[7]
    stds = Data[6]
    
    blocks = np.zeros((len(I_encoded),blocH,blocL))
    for i, blocEnc in enumerate(I_encoded):
        bloc = np.unpackbits(np.array(blocEnc, dtype=np.uint8))[-blockSize:]
        q = (bloc==1).sum()
        pixelBloc = np.zeros(pixelPerBloc)
        for j in range(pixelPerBloc):            
            if bloc[j] == 1:
                pixelBloc[j] = int(np.round(means[i]+stds[i]*np.sqrt((pixelPerBloc-q)/q)))
            else:
                pixelBloc[j] = int(np.round(means[i]-stds[i]*np.sqrt(q/(pixelPerBloc-q))))
        np.clip(pixelBloc, 0, 255, pixelBloc)

        blocks[i] = pixelBloc.reshape((blocH,blocL))

    I_decoded = np.zeros((256,256))
    i=0
    for x in range(0,256,blocH):
            for y in range(0,256,blocL):
                I_decoded[x:x+blocH,y:y+blocL]= blocks[i]
                i+=1

    return I_decoded


