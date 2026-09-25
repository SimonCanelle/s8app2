import numpy as np

def reduce(I_source, LIGNES, COLONNES):
    # Interpolation bilinéaire
    L1, C1 = I_source.shape
    L2 = LIGNES
    C2 = COLONNES

    I_reduced = np.zeros((L2, C2), dtype=np.uint8)

    ratio_lignes = L1/L2
    ratio_colonnes = C1/C2

    for ligne in range(L2):
        for col in range(C2):
            l1 = min(int(np.floor(ligne * ratio_lignes)), L1 - 2)
            c1 = min(int(np.floor(col * ratio_colonnes)), C1 - 2)

            A = I_source[l1,c1]
            B = I_source[l1,c1+1]
            C = I_source[l1+1,c1]
            D = I_source[l1+1,c1+1]

            alpha_1 = col*ratio_colonnes - c1
            alpha_2 = ligne*ratio_lignes - l1

            X = (1-alpha_1)*A + alpha_1*B
            Y = (1-alpha_1)*C + alpha_1*D
            Z = (1-alpha_2)*X + alpha_2*Y

            I_reduced[ligne,col] = np.floor(Z).astype(np.uint8)

    return I_reduced
