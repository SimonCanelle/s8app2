import numpy as np

def BTC_decode(Data):
    blocH = 2
    blocL = 2
    I_metadata_dec = Data[7]
    I_encoded = Data[3]

    nBlocks = len(I_encoded)

    pixelPerBloc = blocH*blocL

    means = I_metadata_dec[0:nBlocks]
    stds = I_metadata_dec[nBlocks:nBlocks*2]
    
    blocks = np.zeros((len(I_encoded),2,2))
    for i, bloc in enumerate(I_encoded):
        q = (bloc==1).sum()
        pixelBloc = np.array([0,0,0,0])
        for j in range(pixelPerBloc):            
            if bloc[j] == 1:
                pixelBloc[j] = means[i]+stds[i]*np.sqrt((pixelPerBloc-q)/q)
            else:
                pixelBloc[j] = means[i]-stds[i]*np.sqrt(q/(pixelPerBloc-q))
        blocks[i] = pixelBloc.reshape((2,2))

    I_decoded = np.zeros((256,256))
    i=0
    for x in range(0,256,2):
            for y in range(0,256,2):
                I_decoded[x:x+2,y:y+2]= blocks[i]
                i+=1

    return I_decoded, I_metadata_dec


