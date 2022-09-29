import numpy as np
import matplotlib.pyplot as plt


def vec_cross(p1, p2, q1, q2):

    v = p2 - p1
    u = q2 - q1

    vxu = np.cross(v, u)

    return vxu


def sigmoid(xv, u, s):

    r = -1.*(xv-u)/s
    k = 1. + np.exp(r)
    p = 1. / k

    return p


# 1D data
def gen_1d_data(x0: float = -5., step: float = 0.1, n_pt: int = 100, f1: float = 0.5):

    xv = []
    yv = []
    for i in range(n_pt):
        xi = x0 + i*step
        yi = sigmoid(xi, 0, f1)

        rand_p = np.random.rand()
        if rand_p > (1 - yi):
            yv.append(1.)
        else:
            yv.append(0.)

        xv.append(xi)

    xa = np.array(xv)
    ya = np.array(yv)
    return ya, xa


# two side data
def gen_2d_line():

    xa = np.arange(410.92, 428.47, 0.28)
    ya = np.arange(235.62, 247.22, 0.5)
    xm, ym = np.meshgrid(xa, ya)
    zm = ym - ym
    ny = xm.shape[0]
    nx = xm.shape[1]

    p1 = np.array([410.92, 241.1, 0])
    p2 = np.array([428.47, 242.3, 0])
    for j in range(ny):
        for i in range(nx):

            q1 = np.array([xm[j][i], ym[j][i], 0])

            uxv = vec_cross(p1, p2, p1, q1)
            d = np.linalg.norm(np.cross(p2-p1, p1-q1))/np.linalg.norm(p2-p1)

            if uxv[2] > 0. and d > 0.1:
                zm[j][i] = np.random.normal(15., 3.)
            elif uxv[2] < 0 and d > 0.1:
                zm[j][i] = np.random.normal(-1., 2)
            else:
                zm[j][i] = np.random.normal(8, 3)

    print(' Data shape = %d , %d' % (zm.shape[0], zm.shape[1]))

    return zm, xm, ym


def gen_2d_arc():

    xa = np.arange(5.0, 65., 2)
    ya = np.arange(10.0, 60., 2)
    xm, ym = np.meshgrid(xa, ya)
    zm = ym - ym
    ny = xm.shape[0]
    nx = xm.shape[1]
    r = 38.
    xc = 65.
    yc = 60.

    for j in range(ny):
        for i in range(nx):

            xi = xm[j][i]
            yi = ym[j][i]
            ri = np.sqrt(np.square(xi-xc) + np.square(yi-yc))
            if (ri-r) > 3.:
                zm[j][i] = np.random.normal(-1, 4)
            elif (ri-r) < 1.:
                zm[j][i] = np.random.normal(15., 4)
            else:
                zm[j][i] = np.random.normal(8, 3)

    return zm, xm, ym


def binarize_2d(zm: np.array, threshold: float = 0.):

    ny = zm.shape[0]
    nx = zm.shape[1]
    bm = zm - zm
    for j in range(ny):
        for i in range(nx):

            if zm[j][i] >= threshold:
                bm[j][i] = 1.
            else:
                bm[j][i] = 0.

    return bm


