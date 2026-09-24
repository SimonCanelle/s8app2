import numpy as np

def interp_bilineaire(I_source, L2, C2):
    L1, C1 = I_source.shape

    I_redim = np.zeros((L2, C2), dtype=np.uint8)

    for l in range(L2):
        for c in range(C2):

            # Equivalent to MATLAB:
            # ll = round((l-1)*(L1-1)/(L2-1)) + 1;
            # cc = round((c-1)*(C1-1)/(C2-1)) + 1;

            ll = int(np.floor(l * (L1 - 1) / (L2 - 1) + 0.5))
            cc = int(np.floor(c * (C1 - 1) / (C2 - 1) + 0.5))

            I_redim[l, c] = I_source[ll, cc]

    return I_redim