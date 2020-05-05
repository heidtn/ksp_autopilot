import cvxpy as cvp
import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt

"""
http://www.larsblackmore.com/iee_tcst13.pdf

carrying the same assumption here that X is up (i.e. normal to land)

"""


GFoldConfig = namedtuple('GFoldConfig', ['isp', 'm', 'm_fuel',
                                         'min_T_p', 'max_T_p',
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


def solve_gfold(config, iterations=450):
    N = iterations

    cost = []
    constr = []

    # Constants
    x_0 = np.array([*config.start_pos, *config.start_vel])
    print(x_0)
    A_w = create_A(config.w)
    n = np.array([1, 0, 0])
    B = create_B()
    g = config.g
    alpha = 1.0 / (config.isp * 9.8)
    pointing_angle = np.cos(config.pointing_lim)
    q = config.q
    v_max = config.v_max
    p1 = config.min_T_p
    p2 = config.max_T_p
    c = get_cone(config.landing_cone)

    dt = 4.5

    # Solved variables
    x = cvp.Variable((6, N))
    u = cvp.Variable((3, N)) 
    gam = cvp.Variable(N)
    z = cvp.Variable(N)

    # initial settings
    #constr += [x[:, 0] == x_0]
    constr += [x[0:3, 0] == x_0[0:3]]
    constr += [x[3:6, 0] == x_0[3:6]]

    constr += [x[3:6, N-1] == np.array([0, 0, 0])]

    #constr += [gam[N-1] == 0]
    #constr += [z[0] == cvp.log(config.m)]
    #constr += [u[:, 0] == gam[0]*_e1]
    #constr += [u[:,N-1] == gam[N-1]*_e1]

    #constr += [_e1@x[0:3,N-1] >= 0]


    for k in range(N-1):
        constr += [x[:, k+1] == x[:, k] + dt*(A_w@x[:, k])]
        constr += [x[3:6, k+1] == x[3:6, k] + dt*(g + u[:, k])]
        #constr += [z[k+1] == z[k] - dt*alpha*gam[k]]

        #constr += [cvp.norm(x[3:6, k]) <= v_max]
        #constr += [cvp.norm(u[:,k]) <= gam[k]]
        #constr += [u[:,k] >= pointing_angle*gam[k]]
        #constr += [cvp.norm(x[0:3,k] - q[:]) - c.T@(x[0:3, k] - q[:])  <= 0 ] # specific, but faster

        if k > 0:
            """
            z_0 = cvp.log(config.m - alpha*p2*k*dt)
            sigma_lower = p1 * cvp.exp(-z_0) * (1 - (z[k] - z_0) + (z[k] - z_0)**2.0/2)
            sigma_upper = p2 * cvp.exp(-z_0) * (1 - (z[k] - z_0))
            constr += [g[k] <= sigma_upper]
            constr += [g[k] >= sigma_lower]
            """
            
            z0_term = config.m - alpha * p2 * (k) * dt  # see ref [2], eq 34,35,36
            z1_term = config.m - alpha * p1 * (k) * dt
            z0 = cvp.log( z0_term )
            z1 = cvp.log( z1_term )
            mu_1 = p1/(z1_term)
            mu_2 = p2/(z0_term)

            # lower thrust bound
            #constr += [gam[k] <= mu_2 * (1 - (z[k] - z0))] # upper thrust bound
            #constr += [z[k] >= z0] # Ensures physical bounds on z are never violated
            #constr += [z[k] <= z1]

    obj = cvp.norm(x[0:3, N-1] - q[:])
    problem = cvp.Problem(cvp.Minimize(obj), constr)
    problem.solve(solver=cvp.ECOS, verbose=True, feastol=5e-15)

    return x


if __name__ == "__main__":
    config = GFoldConfig(isp=203.94,
                         m=(2*1e3 + (0.3)*1e3),
                         m_fuel=(0.3)*1e3,
                         min_T_p=0.2*24000,
                         max_T_p=0.8*24000,
                         g=np.array([-3, 0, 0]),
                         pointing_lim=np.deg2rad(45),
                         landing_cone=np.deg2rad(30),
                         q=np.array([0, 0, 0]),
                         v_max=90,
                         w=np.array([0, 0, 0]),
                         start_pos=np.array([2400, 2000, 0]),
                         start_vel=np.array([-40, 30, 0])
                         )
    x = solve_gfold(config)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.plot(x[0, :].value, x[1, :].value, x[2, :].value)