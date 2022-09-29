from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from DataGenerator import *
import numpy as np
import matplotlib.pyplot as plt

# 1D case
'''
ya, xa = gen_1d_data(x0=-5, step=0.2, n_pt=50, f1=0.66)

Xb = xa.reshape(-1, 1)
print(Xb)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(Xb, ya)

pp = model.predict_proba(Xb)
print(pp)

px = model.predict(Xb)
print(px)
print(ya)

print(' coef = %.5f , b = %.5f, ' % (model.coef_, model.intercept_))

score = model.score(Xb, ya)
print(score)
abcd = confusion_matrix(ya, px)
print(abcd)
'''
# 2D case

zm, xm, ym = gen_2d_line()
#zm, xm, ym = gen_2d_arc()
bm = binarize_2d(zm, 8.)
fm = zm - zm

ny = zm.shape[0]
nx = zm.shape[1]
xy = []
zv = []
for j in range(ny):
    for i in range(nx):

        xy.append([xm[j][i], ym[j][i]])
        zv.append(bm[j][i])

xy1 = np.array(xy)
zv1 = np.array(zv)
print('xv1')
print(xy1.shape)
print(xy1)
print('zv1')
print(zv1.shape)
print(zv1)

# model = LogisticRegression(solver='liblinear', random_state=0)
model = LogisticRegression(solver='newton-cg', random_state=0, tol=0.00001, max_iter=200)
model.fit(xy1, zv1)

pp = model.predict_proba(xy1)
print(pp)
pd = model.predict(xy1)
print('=== predict===')
print(pd)
edge = []
for j in range(ny):
    for i in range(nx):
        k = (j*nx) + i
        # fm[j][i] = pp[k][1]
        fm[j][i] = pd[k]

        if j == 0 and i > 0:
            if pd[k] != pd[k-1]:
                edge.append([xm[j][i], ym[j][i]])
        if j == ny-1 and i > 0:
            if pd[k] != pd[k-1]:
                edge.append([xm[j][i], ym[j][i]])


bx_left = fm[:, :1]
bx_left1 = bx_left[:, 0]
for i in range(ny):
    if i > 0 and bx_left1[i] != bx_left1[i-1]:
        edge.append([xm[i][0], ym[i][0]])

bx_right = fm[:, -1:]
bx_right1 = bx_right[:, 0]
for i in range(ny):
    if i > 0 and bx_right1[i] != bx_right1[i-1]:
        edge.append([xm[i][-1], ym[i][-1]])

x_seg = []
y_seg = []
for it in edge:
    print(' end point (%.2f, %.2f)' % (it[0], it[1]))
    x_seg.append(it[0])
    y_seg.append(it[1])


# 4. Display the results
fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot()
# im = ax.imshow(bm, cmap='rainbow', extent=[xm[0][0], xm[0][-1], ym[0][0], ym[-1][0]], origin='lower')
# plt.cb = fig.colorbar(im, ax=ax)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xm, ym, zm, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.scatter(xm, ym, zm, c='r', marker='o')

fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot()
plt.plot(x_seg, y_seg, 'o', linestyle="-", linewidth=2, color='black')
im1 = ax1.imshow(bm, cmap='rainbow', extent=[xm[0][0], xm[0][-1], ym[0][0], ym[-1][0]], origin='lower')
plt.cb = fig1.colorbar(im1, ax=ax1)

fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot()
im2 = ax2.imshow(fm, cmap='rainbow', extent=[xm[0][0], xm[0][-1], ym[0][0], ym[-1][0]], origin='lower')
plt.cb = fig1.colorbar(im2, ax=ax2)

plt.show()
