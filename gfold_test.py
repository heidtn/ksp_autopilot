import cvxpy as cvp
import numpy as np
from collections import namedtuple

"""
http://www.larsblackmore.com/iee_tcst13.pdf

carrying the same assumption here that X is up (i.e. normal to land)

"""


GFoldConfig = namedtuple('GFoldConfig', ['isp', 'm', 'm_fuel',
                                         'min_T_p', 'max_T_p', 'max_force',
                                         'g', 'pointing_lim', 'landing_cone',
                                         'q', 'v_max', 'w', 'start_pos',
                                         'start_vel'])


_e1 = np.array([1, 0 ,0]).T
_e2 = np.array([0, 1 ,0]).T
_e3 = np.array([0, 0 ,1]).T
_E = np.array([[0, 1, 0], [0, 0, 1]])


def create_S(w):
    S = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    return S

def create_A(w):
    S = create_S(w)
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -np.square(S)
    A[3:6, 3:6] = -2*S
    return A

def create_B():
    B = np.zeros((6, 3))
    B[0:3, :] = np.eye(3)
    return B

def get_cone(cone_angle):
    # cone angle between 0 and pi/2
    return _e1 / np.tan(cone_angle)


def solve_gfold(config, iterations=50):
    N = iterations

    cost = []
    constr = []

    # Constants
    x_0 = np.array([*config.start_pos, *config.start_vel])
    A_w = create_A(config.w)
    n = np.array([1, 0, 0])
    B = create_B()
    g = config.g
    alpha = 1.0 / (config.isp * 9.8)
    pointing_angle = np.cos(config.pointing_lim)

    # Solved variables
    x = cvp.Variable((6, N))
    u = cvp.Variable((3, N))
    gam = cvp.Variable(N)
    m = cvp.Variable(N)
    tf = cvp.Variable(1, value=[5.0])

    # initial settings
    constr += [x[:, 0] == x_0]
    constr += [x[3:6, N-1] == np.array([0, 0, 0])]
    constr += [m[0] == config.m]
    constr += [m[N-1] >= config.m - config.m_fuel]
    constr += [_e1@x[0:3,N-1] == 0]

    for k in range(N-1):
        constr += [x[:, k+1] == x[:, k] + tf*(A_w@x[:, k])]
        constr += [m[k+1] == m[k] - alpha*gam]
        constr += [cvp.norm(u) <= gam]
        constr += [gam >= config.min_T_p * config.max_force]
        constr += [gam <= config.max_T_p * config.max_force]
        #constr += []



if __name__ == "__main__":
    config = GFoldConfig(isp=350,
                         m=12000,
                         m_fuel=5000,
                         min_T_p=0.2,
                         max_T_p=0.8,
                         max_force=250000,
                         g=np.array([-3, 0, 0]),
                         pointing_lim=np.deg2rad(45),
                         landing_cone=np.deg2rad(45),
                         q=np.array([0, 0, 0]),
                         v_max=1500,
                         w=np.array([0, 1e-5, 0]),
                         start_pos=np.array([1e5, 3000, 0]),
                         start_vel=np.array([100, 400, 200])
                         )
    solve_gfold(config)