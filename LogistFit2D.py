from DataGenerator import *
import numpy as np
import matplotlib.pyplot as plt
# import scipy.optimize as optm
# import typing


def sigmoid(x: np.array, w: np.array, b: float):

    z = np.matmul(w, x) + b
    k = 1.0 + np.exp(-z)
    p = 1.0 / k

    return p


# Compute likelihood and gradient at the same time
def likelihood(z: np.array, x: np.array, y: np.array, w: np.array, b: float):

    nx = x.shape[1]
    ny = x.shape[0]
    sum_ln_p = 0.

    mx = 0
    my = 0
    mb = 0
    k = 0
    for j in range(ny):
        for i in range(nx):

            xy = np.array([x[j][i], y[j][i]])
            pi = sigmoid(xy, w, b)
            p = 1.
            znorm = 1.
            if z[j][i] >= 7.5:
                p = pi
            if z[j][i] < 7.5:
                p = 1 - pi
                znorm = 0.

            dx = (pi - znorm)*xy[0]
            dy = (pi - znorm)*xy[1]
            db = (pi - znorm)
            mx = mx + dx
            my = my + dy
            mb = mb + db

            if p > 0.0000001:
                sum_ln_p = sum_ln_p + np.log(p)
                k = k + 1
            else:
                p = 0.0000001
                sum_ln_p = sum_ln_p + np.log(p)

    mx = mx/(nx*ny)
    my = my/(nx*ny)
    mb = mb/(nx*ny)
    if abs(sum_ln_p) < 0.0001:
        print(' LL = %.6f -> %d ' % (sum_ln_p, k))
    # print('Total sum of lnL = %.5f with %d entries' % (sum_ln_p, len(x)))

    return sum_ln_p, mx, my, mb


def gradient(zm: np.array, xm: np.array, ym: np.array, w: np.array, b: float):

    ny = zm.shape[0]
    nx = zm.shape[1]
    mx = 0
    my = 0
    mb = 0
    for j in range(ny):
        for i in range(nx):

            znorm = 1.
            if zm[j][i] < 7.5:
                znorm = 0.

            xy = np.array([xm[j][i], ym[j][i]])
            p = sigmoid(xy, w, b)
            dx = (p - znorm)*xy[0]
            dy = (p - znorm)*xy[1]
            db = (p - znorm)
            mx = mx + dx
            my = my + dy
            mb = mb + db
    mx = mx/(nx*ny)
    my = my/(nx*ny)
    mb = mb/(nx*ny)

    return mx, my, mb


def gradient_search(zm: np.array, xm: np.array, ym: np.array):

    w1 = 5
    w2 = 5.
    b = -100.
    get_min = False
    h1 = 1.
    h2 = 1.
    hb = 0.
    k = 0
    best_w1 = w1
    best_w2 = w2
    best_b = b

    ll_max = -999999999.
    parv = [ll_max, 0, 0, 0]
    forward = True
    hs = 0.1
    while get_min is False:
        w = np.array([w1, w2])
        ll, gx, gy, gb = likelihood(zm, xm, ym, w, b)

        if ll < parv[0] and forward is True:
            print('(%d) w,b=(%.3f, %.3f, %.2f), grad = (%.4f, %.4f, %3f), h: %.3f, ll = %.3f' % (k, w1, w2, b, gx, gy, gb, hs, ll))
            w1 = w1 + (parv[1]*h1*0.9)
            w2 = w2 + (parv[2]*h2*0.9)
            b = b + (parv[3]*hb*0.9)
            forward = False
            hs = hs*0.6
        else:
            print('[%d] w,b=(%.3f, %.3f, %.2f), grad = (%.4f, %.4f, %3f), h: %.3f, ll = %.3f' % (k, w1, w2, b, gx, gy, gb, hs, ll))
            if abs(gx/w1) > hs:
                h1 = hs*abs(w1/gx)
            if abs(gy/w2) > hs:
                h2 = hs*abs(w2/gy)
            if abs(gb/b) > hs:
                hb = hs*abs(b/gb)

            hb = 0.
            w1 = w1 - (gx*h1)
            w2 = w2 - (gy*h2)
            b = b - (gb*hb)
            parv = [ll, gx, gy, gb]
            forward = True

        k = k + 1
        dll = ll - ll_max
        # if dll < 0.001 and abs(gx) < 0.005 and abs(gy) < 0.005 and abs(gb) < 0.005:
        if dll < 0.001 and ll_max > -10.0:
        # if dll < 0.001 and gx < 0.01 and gy < 0.01 and gb < 0.01:
            get_min = True
            print('[%d] w,b=(%.3f, %.3f, %.2f), grad = (%.4f, %.4f, %3f), ll = %.3f' % (k, w1, w2, b, gx, gy, gb, ll))

        if ll > ll_max:
            ll_max = ll
            best_w1 = w1
            best_w2 = w2
            best_b = b

        if k == 100000:
            get_min = True

    return best_w1, best_w2, best_b


# 1. Generate data
print(' ===== ')
zm, xm, ym = gen_2d_line()
# zm, xm, ym = gen_2d_arc()
ny = zm.shape[0]
nx = zm.shape[1]

# 2. Gradient Search for best parameters
best_w1, best_w2, best_b = gradient_search(zm, xm, ym)
best_w = [best_w1, best_w2]

# 3. Show fitted data/result
zm_fit = ym - ym
for j in range(ny):
    for i in range(nx):
        pij = np.array([xm[j][i], ym[j][i]])
        wij = np.array(best_w)
        zm_fit[j][i] = sigmoid(pij, wij, best_b)

# 4. Display the results
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
# ax = plt.subplot2grid((2, 1), (0, 0))
ax.plot_surface(xm, ym, zm, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.scatter(xm, ym, zm, c='r', marker='o')

fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot()
im = ax1.imshow(zm_fit, cmap='rainbow', extent=[xm[0][0], xm[0][-1], ym[0][0], ym[-1][0]], origin='lower')
plt.cb = fig1.colorbar(im, ax=ax1)

plt.show()
