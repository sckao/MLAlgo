import numpy as np
import scipy.spatial.transform as transform


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


def length(p1, p2):

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    the_len = np.sqrt(np.square(x1-x2) + np.square(y1-y2))

    return the_len


def transform_coord(pos, theta: float, sx: float, sy: float):

    r_mtx = transform.Rotation.from_rotvec([0., 0, theta], degrees=True)
    # print(r_mtx.as_matrix())
    rm = r_mtx.as_matrix()

    p1 = [pos[0], pos[1], 0]

    shift_xy = np.array([sx, sy, 0])
    np1 = np.matmul(rm, p1) + shift_xy

    return np1


def distance_line(edge_p1, edge_p2, pos):

    edge_vx = edge_p2[0] - edge_p1[0]
    edge_vy = edge_p2[1] - edge_p1[1]
    v1 = np.array([edge_vx, edge_vy])
    m1 = np.linalg.norm(v1)

    px = pos[0] - edge_p1[0]
    py = pos[1] - edge_p1[1]
    v2 = np.array([px, py])

    axb = np.cross(v1, v2)
    m_axb = np.linalg.norm(axb)

    d = abs(m_axb/m1)

    return d


np1 = transform_coord([5, 3], 38, 2.5, 1.3)
np2 = transform_coord([5, 8], 38, 2.5, 1.3)
np3 = transform_coord([9, 11], 38, 2.5, 1.3)
np4 = transform_coord([21, 11], 38, 2.5, 1.3)

print('P12 (%.3f, %.3f) - (%.3f, %.3f) ' % (np1[0], np1[1], np2[0], np2[1]))
print('P34 (%.3f, %.3f) - (%.3f, %.3f) ' % (np3[0], np3[1], np4[0], np4[1]))
