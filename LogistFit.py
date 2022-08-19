import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm
from sklearn.linear_model import LogisticRegression


def log_fit(xv, u, s, a, b):

    p = a / (1. + np.exp((xv-u)/-s)) + b
    return p


def log_fun(xv, u, s):

    r = -1.*(xv-u)/s
    k = 1. + np.exp(r)
    p = 1. / k

    return p


def likelihood(x, y, u, s):

    sum_ln_p = 0.

    for i in range(len(x)):

        pi = log_fun(x[i], u, s)
        p = 1.
        if y[i] > 0.999:
            p = pi
        if y[i] < 0.001:
            p = 1 - pi

        if p != 0.0:
            sum_ln_p = sum_ln_p + np.log(p)

    # print('Total sum of lnL = %.5f with %d entries' % (sum_ln_p, len(x)))

    return sum_ln_p


x = []
y = []
Z = []
w = []
step = 0.1
x0 = -5.
for i in range(100):
    xi = x0 + i*step
    yi = log_fun(xi, 0, 0.2)
    zi = log_fun(xi, 0, 0.8)

    wi = (yi + zi) / 2.
    rand_p = np.random.rand()
    if rand_p > (1 - wi):
        w.append(1.)
    else:
        w.append(0.)

    x.append(xi)
    y.append(yi)
    Z.append(zi)

xa = np.array(x)
wa = np.array(w)
par, cov = optm.curve_fit(log_fit, xa, wa)
print(' Fit para')
print(par)
print(' Fit cov')
print(cov)

ll_max = -9999999.
best_fit = 0.
best_ctr = 0.
for k in range(10):
    for j in range(10):

        u = -0.5 + (j*0.1)
        s = k*0.1 + 0.05
        ll = likelihood(xa, wa, u, s)
        if ll > ll_max:
            ll_max = ll
            best_fit = s
            best_ctr = u

print('best fit @ %.3f = %.2f , %.4f ' % (best_ctr, best_fit, ll_max))
w_fit = log_fun(xa, 0., best_fit)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'r')
plt.plot(x, Z, 'b')
plt.plot(x, w_fit, 'g')
plt.scatter(x, w, marker='^')
plt.ylabel('Y')
plt.xlabel('X')

plt.show()