import numpy as np
from numba import jit

@jit(nopython=True)
def sumNN(Ri,Rj,S):
    sumRiRj = np.zeros(shape=(len(Ri),len(Rj)))
    Sx,Sy,Sz = S
    for i in range(len(Ri)):
        for j in range(len(Rj)):
            # 0 axis
            if i+1 < len(Ri):
                ip1 = i+1
            else:
                ip1 = i+1-len(Ri)
            if i-1>=0:
                im1 = i-1
            else:
                im1 = i-1+len(Ri)
            # 1 axis
            if j+1 < len(Ri):
                jp1 = j+1
            else:
                jp1 = j+1-len(Ri)
            if j-1>=0:
                jm1 = j-1
            else:
                jm1 = j-1+len(Ri)
            sumRiRj[i][j] = Sx[i][j]*Sx[ip1][j] + Sy[i][j]*Sy[ip1][j] + Sz[i][j]*Sz[ip1][j] + \
                            Sx[i][j]*Sx[ip1][jp1] + Sy[i][j]*Sy[ip1][jp1] + Sz[i][j]*Sz[ip1][jp1] + \
                            Sx[i][j]*Sx[i][jp1] + Sy[i][j]*Sy[i][jp1] + Sz[i][j]*Sz[i][jp1] + \
                            Sx[i][j]*Sx[im1][j] + Sy[i][j]*Sy[im1][j] + Sz[i][j]*Sz[im1][j] + \
                            Sx[i][j]*Sx[im1][jm1] + Sy[i][j]*Sy[im1][jm1] + Sz[i][j]*Sz[im1][jm1] + \
                            Sx[i][j]*Sx[i][jm1] + Sy[i][j]*Sy[i][jm1] + Sz[i][j]*Sz[i][jm1]
    return sumRiRj


@jit(nopython=True)
def sumNNN(Ri,Rj,S):
    sumRiRj = np.zeros(shape=(len(Ri),len(Rj)))
    Sx,Sy,Sz = S
    for i in range(len(Ri)):
        for j in range(len(Rj)):
            # 0 axis
            if i+1 < len(Ri):
                ip1 = i+1
            else:
                ip1 = i+1-len(Ri)
            if i-1>=0:
                im1 = i-1
            else:
                im1 = i-1+len(Ri)
            # 1 axis
            if j+1 < len(Ri):
                jp1 = j+1
            else:
                jp1 = j+1-len(Ri)
            if j-1>=0:
                jm1 = j-1
            else:
                jm1 = j-1+len(Ri)
            # 0 axis
            if i+2 < len(Ri):
                ip2 = i+2
            else:
                ip2 = i+2-len(Ri)
            if i-2>=0:
                im2 = i-2
            else:
                im2 = i-2+len(Ri)
            # 1 axis
            if j+2 < len(Ri):
                jp2 = j+2
            else:
                jp2 = j+2-len(Ri)
            if j-2>=0:
                jm2 = j-2
            else:
                jm2 = j-2+len(Ri)
            sumRiRj[i][j] = Sx[i][j]*Sx[ip2][jp1] + Sy[i][j]*Sy[ip2][jp1] + Sz[i][j]*Sz[ip2][jp1] + \
                            Sx[i][j]*Sx[ip1][jp2] + Sy[i][j]*Sy[ip1][jp2] + Sz[i][j]*Sz[ip1][jp2] + \
                            Sx[i][j]*Sx[im1][jp1] + Sy[i][j]*Sy[im1][jp1] + Sz[i][j]*Sz[im1][jp1] + \
                            Sx[i][j]*Sx[im2][jm1] + Sy[i][j]*Sy[im2][jm1] + Sz[i][j]*Sz[im2][jm1] + \
                            Sx[i][j]*Sx[im1][jm2] + Sy[i][j]*Sy[im1][jm2] + Sz[i][j]*Sz[im1][jm2] + \
                            Sx[i][j]*Sx[ip1][jm1] + Sy[i][j]*Sy[ip1][jm1] + Sz[i][j]*Sz[ip1][jm1]

    return sumRiRj


# @jit(nopython=True)
# def sumNNN(Ri, Rj, S):
#     sumRiRj = np.zeros(shape=(len(Ri), len(Rj)))
#     Sx, Sy, Sz = S
#     N = len(Ri)  # Number of elements in the array

#     for i in range(len(Ri)):
#         for j in range(len(Rj)):
#             # Calculate wrapped indices for the boundaries
#             ip1 = (i + 1) % N
#             ip2 = (i + 2) % N
#             im1 = (i - 1) % N
#             im2 = (i - 2) % N
#             jp1 = (j + 1) % N
#             jp2 = (j + 2) % N
#             jm1 = (j - 1) % N
#             jm2 = (j - 2) % N

#             sumRiRj[i][j] = (
#                 Sx[i][j]*Sx[ip2][jp1] + Sy[i][j]*Sy[ip2][jp1] + Sz[i][j]*Sz[ip2][jp1] +
#                 Sx[i][j]*Sx[ip1][jp2] + Sy[i][j]*Sy[ip1][jp2] + Sz[i][j]*Sz[ip1][jp2] +
#                 Sx[i][j]*Sx[im1][jp1] + Sy[i][j]*Sy[im1][jp1] + Sz[i][j]*Sz[im1][jp1] +
#                 Sx[i][j]*Sx[im2][jm1] + Sy[i][j]*Sy[im2][jm1] + Sz[i][j]*Sz[im2][jm1] +
#                 Sx[i][j]*Sx[im1][jm2] + Sy[i][j]*Sy[im1][jm2] + Sz[i][j]*Sz[im1][jm2] +
#                 Sx[i][j]*Sx[ip1][jm1] + Sy[i][j]*Sy[ip1][jm1] + Sz[i][j]*Sz[ip1][jm1]
#             )

#     return sumRiRj

# @jit(nopython=True)
# def sumNN(Ri, Rj, S):
#     sumRiRj = np.zeros(shape=(len(Ri), len(Rj)))
#     Sx, Sy, Sz = S
#     N = len(Ri)  # Number of elements in the array

#     for i in range(len(Ri)):
#         for j in range(len(Rj)):
#             # Calculate wrapped indices for the boundaries
#             ip1 = (i + 1) % N
#             im1 = (i - 1) % N
#             jp1 = (j + 1) % N
#             jm1 = (j - 1) % N

#             sumRiRj[i][j] = Sx[i][j]*Sx[ip1][j] + Sy[i][j]*Sy[ip1][j] + Sz[i][j]*Sz[ip1][j] + \
#                             Sx[i][j]*Sx[ip1][jp1] + Sy[i][j]*Sy[ip1][jp1] + Sz[i][j]*Sz[ip1][jp1] + \
#                             Sx[i][j]*Sx[i][jp1] + Sy[i][j]*Sy[i][jp1] + Sz[i][j]*Sz[i][jp1] + \
#                             Sx[i][j]*Sx[im1][j] + Sy[i][j]*Sy[im1][j] + Sz[i][j]*Sz[im1][j] + \
#                             Sx[i][j]*Sx[im1][jm1] + Sy[i][j]*Sy[im1][jm1] + Sz[i][j]*Sz[im1][jm1] + \
#                             Sx[i][j]*Sx[i][jm1] + Sy[i][j]*Sy[i][jm1] + Sz[i][j]*Sz[i][jm1]

#     return sumRiRj
