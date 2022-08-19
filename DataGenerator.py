import numpy as np
import matplotlib.pyplot as plt


def vec_cross(p1, p2, q1, q2):

    v = p2 - p1
    u = q2 - q1

    vxu = np.cross(v, u)

    return vxu


# two side data
def gen_2d_line():

    xa = np.arange(5.0, 65., 3)
    ya = np.arange(10.0, 60., 5)
    xm, ym = np.meshgrid(xa, ya)
    zm = ym - ym
    ny = xm.shape[0]
    nx = xm.shape[1]

    p1 = np.array([5, 41, 0])
    p2 = np.array([65, 23, 0])
    for j in range(ny):
        for i in range(nx):

            q1 = np.array([xm[j][i], ym[j][i], 0])

            uxv = vec_cross(p1, p2, p1, q1)

            if uxv[2] > 0:
                zm[j][i] = np.random.normal(15., 2)
            if uxv[2] < 0:
                zm[j][i] = np.random.normal(-1, 2)
            if uxv[2] == 0:
                zm[j][i] = np.random.normal(8, 7)

    print(' Data shape = %d , %d' % (zm.shape[0], zm.shape[1]))

    return zm, xm, ym


def gen_2d_arc():

    xa = np.arange(5.0, 65., 3)
    ya = np.arange(10.0, 60., 5)
    xm, ym = np.meshgrid(xa, ya)
    zm = ym - ym
    ny = xm.shape[0]
    nx = xm.shape[1]
    r = 42.
    xc = 75.
    yc = 30.

    for j in range(ny):
        for i in range(nx):

            xi = xm[j][i]
            yi = ym[j][i]
            ri = np.sqrt(np.square(xi-xc) + np.square(yi-yc))
            if (ri-r) > 1.:
                zm[j][i] = np.random.normal(-1, 2)
            elif (ri-r) < 1.:
                zm[j][i] = np.random.normal(15., 2)
            else:
                zm[j][i] = np.random.normal(8, 7)

    return zm, xm, ym
