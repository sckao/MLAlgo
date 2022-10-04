# Testing method to justify a point is on which side of the vector

import numpy as np


def vec_cross(p1, p2, q1, q2):

    v = p2 - p1
    u = q2 - q1

    vxu = np.cross(v, u)

    return vxu


# test
xa = np.arange(10, 40, 2)
ya = np.arange(5, 35, 5)
print(xa.shape)
print(ya.shape)
# print(' xa sz = %d , ya sz = %d' %(xa.size(), ya.size()))
xm, ym = np.meshgrid(xa, ya)
zm = xm - xm
print(xm.shape)
print(xm)
print(ym)
print(' xm slicing')
xm1 = xm[1::, 2:10]
ym1 = ym[1::, 2:10]
print(xm1)
print(' ym slicing')
print(ym1)
print(' ============= ')

p1 = np.array([xm[-1][0], ym[-1][0], 0])
p2 = np.array([xm[0][-1], ym[0][-1], 0])
print(p1)
print(p2)

ny = xm.shape[0]
nx = xm.shape[1]

for j in range(ny):
    for i in range(nx):

        q1 = np.array([xm[j][i], ym[j][i], 0])

        uxv = vec_cross(p1, p2, p1, q1)

        if uxv[2] > 0:
            zm[j][i] = np.random.normal(15., 2)
        if uxv[2] < 0:
            zm[j][i] = np.random.normal(-5, 2)
        if uxv[2] == 0:
            zm[j][i] = np.random.normal(10, 1)


print(zm)
