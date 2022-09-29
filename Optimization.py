import scipy.optimize as optm
import scipy.spatial.transform as transform
# from scipy.spatial.transform import Rotation as R
import PointGeometry as geo
import numpy as np
import matplotlib.pyplot as plt


class Align:

    __instance__ = None
    align_r = -1
    align_d = 999999

    def __init__(self):
        Align.__instance__ = self

    @staticmethod
    def get_instance():
        if Align.__instance__ is None:
            Align()
        # end if
        return Align.__instance__

    @staticmethod
    def collinear(p1, p2, p3):

        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]

        val = ((y2-y1)*(x3-x2)) - ((y3-y2)*(x2-x1))
        # if val = 0 : Collinear
        #    val > 0 : CounterClockwise (Left turn)
        #    val < 0 : Clockwise (Right turn)

        # This criterion fix precision issue
        if abs(val) < 0.0001:
            val = 0.

        return val

    @ staticmethod
    def transform_segment(params, e1x, e1y, e2x, e2y, p1x, p1y, p2x, p2y):

        sx, sy, theta = params
        p1 = np.array([p1x, p1y, 0.])
        p2 = np.array([p2x, p2y, 0.])
        e1 = np.array([e1x, e1y, 0.])
        e2 = np.array([e2x, e2y, 0.])
        r_mtx = transform.Rotation.from_rotvec([0., 0, theta], degrees=True)
        # print(r_mtx.as_matrix())
        rm = r_mtx.as_matrix()

        shift_xy = np.array([sx, sy, 0])
        np1 = np.matmul(rm, p1) + shift_xy
        np2 = np.matmul(rm, p2) + shift_xy

        cl1 = Align.collinear(e1, e2, np1)
        cl2 = Align.collinear(e1, e2, np2)
        cl = np.sqrt((cl1*cl1) + (cl2*cl2))

        len1 = geo.length(p1, p2)
        len2 = geo.length(np1, np2)
        print('fitted y = %.5f --> (%.2f, %.2f)  - (%.2f, %.2f)' % (cl, np1[0], np1[1], np2[0], np2[1]))
        print(' L1 = %.3f , L2 = %.3f' % (len1, len2))
        Align.align_r = cl

        return cl

    @staticmethod
    def fit_segments(params, edge_p1, edge_p2, seg_p1, seg_p2):

        sx, sy = params
        d_sqr_sum = 0.
        for i in range(len(seg_p1)):
            px = seg_p1[i][0] + sx
            py = seg_p1[i][1] + sy

            qx = seg_p2[i][0] + sx
            qy = seg_p2[i][1] + sy

            di = geo.distance_line(edge_p1[i], edge_p2[i], [px, py])
            dj = geo.distance_line(edge_p1[i], edge_p2[i], [qx, qy])

            print('[%d] sxy = (%.2f, %.2f) di = %.3f , dj = %.3f' %(i, sx, sy, di, dj))
            print('      edge = (%.3f, %.3f) - (%.3f, %.3f) : (%.3f, %.3f)'
                  % (edge_p1[i][0], edge_p1[i][1], edge_p2[i][0], edge_p2[i][1], px, py))

            if di < 0.0001:
                di = 0.

            d_sqr_sum = d_sqr_sum + (di*di)

        d_total = np.sqrt(d_sqr_sum)
        print(' ### d sum = %.3f' % d_total)

        Align.align_d = d_total
        return d_total

    def alignment(self, edge_p1, edge_p2, segment_p1, segment_p2):

        e1x = edge_p1[0]
        e1y = edge_p1[1]
        e2x = edge_p2[0]
        e2y = edge_p2[1]
        p1x = segment_p1[0]
        p1y = segment_p1[1]
        p2x = segment_p2[0]
        p2y = segment_p2[1]

        e_xc = (e1x + e2x) / 2.
        e_yc = (e1y + e2y) / 2.
        p_xc = (p1x + p2x) / 2.
        p_yc = (p1y + p2y) / 2.
        sx0 = e_xc - p_xc
        sy0 = e_yc - p_yc
        result = optm.minimize(self.transform_segment,
                               np.array([sx0, sy0, 0.]),
                               args=(e1x, e1y, e2x, e2y, p1x, p1y, p2x, p2y),
                               # method='Powell',
                               # tol=0.001
                               )

        if result.success:
            fitted_params = result.x
            print(' Fitted Result === ')
            print(fitted_params)
            print(' Best align d = %.6f' % self.align_d)
        else:
            fitted_params = result.x
            print(' Failed Fitted Result === ')
            print(fitted_params)
            raise ValueError(result.message)

        return fitted_params, self.align_r

    def global_align(self, edge1, edge2, seg1, seg2):

        sx0 = 0
        sy0 = 0
        result = optm.minimize(self.fit_segments,
                               np.array([sx0, sy0]),
                               args=(edge1, edge2, seg1, seg2),
                               )

        if result.success:
            fitted_params = result.x
            print(' Fitted Result === ')
            print(fitted_params)
        else:
            fitted_params = result.x
            print(' Failed Fitted Result === ')
            print(fitted_params)
            raise ValueError(result.message)

        return fitted_params, self.align_d


np1 = geo.transform_coord([5, 3], 38, 2.5, 1.3)
np2 = geo.transform_coord([5, 8], 38, 2.5, 1.3)
np3 = geo.transform_coord([9, 11], 38, 2.5, 1.3)
np4 = geo.transform_coord([21, 11], 38, 2.5, 1.3)


ep1 = geo.transform_coord([5, 3], 9, 2., 0.3)
ep2 = geo.transform_coord([5, 8], 9, 2., 0.3)
ep3 = geo.transform_coord([9, 11], 9, 2., 0.3)
ep4 = geo.transform_coord([21, 11], 9, 2., 0.3)

alg = Align()
par1, r1 = alg.alignment(ep1, ep2, np1, np2)
par2, r2 = alg.alignment(ep3, ep4, np3, np4)

print('--------------------------------')
print('P12 (%.3f, %.3f) - (%.3f, %.3f) ' % (np1[0], np1[1], np2[0], np2[1]))
print('P34 (%.3f, %.3f) - (%.3f, %.3f) ' % (np3[0], np3[1], np4[0], np4[1]))
print('--------------------------------')
print('E12 (%.3f, %.3f) - (%.3f, %.3f) ' % (ep1[0], ep1[1], ep2[0], ep2[1]))
print('E34 (%.3f, %.3f) - (%.3f, %.3f) ' % (ep3[0], ep3[1], ep4[0], ep4[1]))
print('--------------------------------')

mp1 = geo.transform_coord(np1, par1[2], par1[0], par2[1])
mp2 = geo.transform_coord(np2, par1[2], par1[0], par2[1])
mp3 = geo.transform_coord(np3, par1[2], par1[0], par2[1])
mp4 = geo.transform_coord(np4, par1[2], par1[0], par2[1])
print('M12 (%.3f, %.3f) - (%.3f, %.3f) ' % (mp1[0], mp1[1], mp2[0], mp2[1]))
print('M34 (%.3f, %.3f) - (%.3f, %.3f) ' % (mp3[0], mp3[1], mp4[0], mp4[1]))

d1 = geo.distance_line(ep1, ep2, mp1)
d2 = geo.distance_line(ep1, ep2, mp2)
d3 = geo.distance_line(ep3, ep4, mp3)
d4 = geo.distance_line(ep3, ep4, mp4)
print('======================')
print('d = %.4f, %.4f, %.4f, %.4f' % (d1, d2, d3, d4))
print('-------------')

edge_v1 = [ep1, ep3]
edge_v2 = [ep2, ep4]
seg_v1 = [mp1, mp3]
seg_v2 = [mp2, mp4]
par_g, dg = alg.global_align(edge_v1, edge_v2, seg_v1, seg_v2)

gp1 = geo.transform_coord(mp1, 0, par_g[0], par_g[1])
gp2 = geo.transform_coord(mp2, 0, par_g[0], par_g[1])
gp3 = geo.transform_coord(mp3, 0, par_g[0], par_g[1])
gp4 = geo.transform_coord(mp4, 0, par_g[0], par_g[1])
print('G12 (%.3f, %.3f) - (%.3f, %.3f) ' % (gp1[0], gp1[1], gp2[0], gp2[1]))
print('G34 (%.3f, %.3f) - (%.3f, %.3f) ' % (gp3[0], gp3[1], gp4[0], gp4[1]))

print('-------------')
print(par1)
print('-------------')
print(par2)
print('-------------')

edge_x = [ep1[0], ep2[0], ep3[0], ep4[0]]
edge_y = [ep1[1], ep2[1], ep3[1], ep4[1]]

seg_x = [np1[0], np2[0], np3[0], np4[0]]
seg_y = [np1[1], np2[1], np3[1], np4[1]]

alg_x = [mp1[0], mp2[0], mp3[0], mp4[0]]
alg_y = [mp1[1], mp2[1], mp3[1], mp4[1]]

glb_x = [gp1[0], gp2[0], gp3[0], gp4[0]]
glb_y = [gp1[1], gp2[1], gp3[1], gp4[1]]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot()
plt.plot(edge_x, edge_y, 'o', linestyle="-", linewidth=4, color='black')
plt.plot(seg_x, seg_y, 'o', linestyle="-", linewidth=2, color='red')
# plt.plot(alg_x, alg_y, 'o', linestyle="-", linewidth=2, color='blue')
plt.plot(glb_x, glb_y, 'o', linestyle="-", linewidth=2, color='green')

plt.show()

'''
pos1 = np.array([5, 7, 0])
pos2 = np.array([2, 2.3, 0])


# def alignment(sx: float, sy: float, theta: float):
def alignment(params):

    sx, sy, theta = params

    r_mtx = transform.Rotation.from_rotvec([0., 0, theta], degrees=True)
    print(r_mtx.as_matrix())
    rm = r_mtx.as_matrix()

    shift_xy = np.array([sx, sy, 0])
    np1 = np.matmul(rm, pos1) + shift_xy
    np2 = np.matmul(rm, pos2) + shift_xy

    edge_p1 = [10., 7.2]
    edge_p2 = [10., 11.6]
    cl1 = geo.collinear(edge_p1, edge_p2, np1)
    cl2 = geo.collinear(edge_p1, edge_p2, np2)
    cl = np.sqrt((cl1*cl1) + (cl2*cl2))

    len1 = geo.length(pos1, pos2)
    len2 = geo.length(np1, np2)
    print('fitted y = %.5f --> (%.2f, %.2f)  - (%.2f, %.2f)' % (cl, np1[0], np1[1], np2[0], np2[1]))
    print(' L1 = %.3f , L2 = %.3f' %(len1, len2))

    return cl


result = optm.minimize(alignment, [0., 0., 0.])

if result.success:
    fitted_params = result.x
    print(' Fitted Result === ')
    print(fitted_params)
else:
    raise ValueError(result.message)
'''



