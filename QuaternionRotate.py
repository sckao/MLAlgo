import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# theta in degree
# input v is 3-vector: [v_i, v_j, v_k]
def rotate_matrix(v: np.array, theta_deg):

    theta_rad = theta_deg*np.pi/(180*2)
    u = v / np.linalg.norm(v)
    s = np.sin(theta_rad)

    # unit 4-vector of q
    q = np.array([np.cos(theta_rad), s*u[0], s*u[1], s*u[2]])
    print(' Unit q vect = ')
    print(q)

    c00 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    c01 = 2*(q[1]*q[2] - q[0]*q[3])
    c02 = 2*(q[1]*q[3] + q[0]*q[2])

    c10 = 2*(q[1]*q[2] + q[0]*q[3])
    c11 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    c12 = 2*(q[2]*q[3] - q[0]*q[1])

    c20 = 2*(q[1]*q[3] - q[0]*q[2])
    c21 = 2*(q[2]*q[3] + q[0]*q[1])
    c22 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    rot_mat = np.array([
        [c00, c01, c02],
        [c10, c11, c12],
        [c20, c21, c22]
    ])

    return rot_mat


# input q, r are 3-vector
def rotate(q: np.array, r: np.array, theta_deg: float):

    r_mat = rotate_matrix(q, theta_deg)

    rr = np.matmul(r_mat, r)
    print(' maxtrix multiplication')
    print(r_mat)
    print(r)

    return rr


class Quaternion:

    def __init__(self):

        self.origin = np.array([0., 0., 0.])
        self.axis = np.array([0, 0, 0])

    # vector is defined as v = p1 - p0
    def set_rotate_axis(self, p0: np.array, p1: np.array):

        q = p1 - p0
        qu = q / np.linalg.norm(q)
        self.origin = p0
        self.axis = [qu[0], qu[1], qu[2]]
        return qu

    def rotate_a(self, p0: np.array, p1: np.array, theta_deg: float):

        v0 = p0 - self.origin
        v1 = p1 - self.origin
        rv0 = self.rotate(self.axis, v0, theta_deg)
        rv1 = self.rotate(self.axis, v1, theta_deg)

        rv_0 = rv0 + self.origin
        rv_1 = rv1 + self.origin

        return rv_0, rv_1

    # rotate v3 w.r.t. q3,
    # v3 , q3 are 3-vector: [v_i, v_j, v_k]
    def rotate(self, q3: np.array, v3: np.array, angle_deg: float):

        angle_rad = angle_deg*np.pi/(180*2)
        ca = np.cos(angle_rad)
        sa = np.sin(angle_rad)
        uq = q3 / np.linalg.norm(q3)
        qq = np.array([ca, uq[0]*sa, uq[1]*sa, uq[2]*sa])
        v4 = np.array([0, v3[0], v3[1], v3[2]])
        rot_v = self.do_rotate(qq, v4)

        rot_v3 = np.array([rot_v[1], rot_v[2], rot_v[3]])
        mag = np.linalg.norm(rot_v3)
        print(' Raw     Vec   = %.3f, %.3f, %.3f, %.3f ' % (v4[0], v4[1], v4[2], v4[3]))
        print(' Rotation Axis = %.3f, %.3f, %.3f, %.3f ' % (qq[0], qq[1], qq[2], qq[3]))
        print(' Rotated Vec   =       %.3f, %.3f, %.3f = %.3f ' % (rot_v3[0], rot_v3[1], rot_v3[2], mag))

        return rot_v3

    @staticmethod
    def multiple_matrix(v: np.array):

        vm = np.array([[v[0], -v[1], -v[2], -v[3]],
                       [v[1], v[0], -v[3], v[2]],
                       [v[2], v[3], v[0], -v[1]],
                       [v[3], -v[2], v[1], v[0]]])

        return vm

    # rotation of v = qvq* , q and v are 4-vector [v0, v_i, v_j, v_k]
    def do_rotate(self, qr: np.array, v: np.array):

        vm = self.multiple_matrix(v)
        qj = np.array([qr[0], -qr[1], -qr[2], -qr[3]])
        vq = np.matmul(vm, qj)

        qi = np.array([qr[0], qr[1], qr[2], qr[3]])
        qm = self.multiple_matrix(qi)
        rot_v = np.matmul(qm, vq)

        return rot_v


###########
# testing
###########
v1 = np.array([5, 5, 5])
v0 = np.array([2, 0, 1])
qv = np.array([3, 0, 8])
qv0 = np.array([1, 0, 3])
r_angle_deg = 10
vlist = [[qv0, qv, 'r'], [v0, v1, 'b']]

qtr = Quaternion()
rv1 = qtr.rotate(qv, v1, r_angle_deg)
qtr.set_rotate_axis(qv0, qv)

for i in range(35):

    r0, r1 = qtr.rotate_a(v0, v1, r_angle_deg)
    vlist.append([r0, r1, 'g'])
    r10 = r1 - r0
    v10 = v1 - v0
    m_v10 = np.linalg.norm(v10)
    m_r10 = np.linalg.norm(r10)
    print(' [%d] mag for %.3f - %.3f' % (i,m_v10, m_r10))
    r_angle_deg = r_angle_deg + 10

# 4. Display the results
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

for it in vlist:

    xa = it[0][0]
    ya = it[0][1]
    za = it[0][2]
    ua = it[1][0]-it[0][0]
    va = it[1][1]-it[0][1]
    wa = it[1][2]-it[0][2]
    ax.quiver(xa, ya, za, ua, va, wa, color=it[2])


# ua = [qv[0]-qv0[0], v1[0]-v0[0], r1[0]-r0[0]]
# va = [qv[1]-qv0[1], v1[1]-v0[1], r1[1]-r0[1]]
# wa = [qv[2]-qv0[2], v1[2]-v0[2], r1[2]-r0[2]]
# xa = [qv0[0], v0[0], r0[0]]
# ya = [qv0[1], v0[1], r0[1]]
# za = [qv0[2], v0[2], r0[2]]

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-5, 10])
plt.show()

'''
soa = np.array([[0, 0, 1, 1, -2, 0], [0, 0, 2, 1, 1, 0],
               [0, 0, 3, 2, 1, 0], [0, 0, 4, 0.5, 0.7, 0]])

X, Y, Z, U, V, W = zip(*soa)
ax.quiver(X, Y, Z, U, V, W)
plt.show()
'''