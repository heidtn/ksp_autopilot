import cvxpy as cvp
import numpy as np
from collections import namedtuple

from mayavi import mlab
from mayavi.mlab import points3d, plot3d, quiver3d
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


def solve_gfold(config, iterations=100):
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

    t_max = config.m_fuel / (alpha * p1)
    t_min = (config.m - config.m_fuel) * np.linalg.norm(config.start_vel) / p2

    dt = 1.1

    # Solved variables
    x = cvp.Variable((6, N))
    u = cvp.Variable((3, N)) 
    gam = cvp.Variable(N)
    z = cvp.Variable(N)

    # initial settings
    constr += [x[:, 0] == x_0[:]]
    constr += [x[3:6, N-1] == np.array([0, 0, 0])]

    constr += [gam[N-1] == 0]
    constr += [u[:, 0] == gam[0]*_e1]
    constr += [u[:, N-1] == gam[N-1]*_e1]
    constr += [z[0] == cvp.log(config.m)]

    constr += [x[0, N-1] == 0]
    constr += [x[0, 0:N-1] >= 0]

    for k in range(N-1):
        constr += [x[0:3, k+1] == x[0:3, k] + dt*(A_w@x[:, k])[0:3]]
        constr += [x[3:6, k+1] == x[3:6, k] + dt*(g + u[:, k])]
        constr += [z[k+1] == z[k] - dt*alpha*gam[k]]

        constr += [cvp.norm(x[3:6, k]) <= v_max]
        constr += [cvp.norm(u[:,k]) <= gam[k]]
        constr += [u[0,k] >= pointing_angle*gam[k]]
        constr += [cvp.norm(x[0:3,k] - q[:]) - c.T@(x[0:3, k] - q[:])  <= 0 ]

        if k > 0:
            z_0 = cvp.log(config.m - alpha * p2 * (k) * dt)
            z_1 = cvp.log(config.m - alpha * p1 * (k) * dt)

            sigma_lower = p1 * cvp.exp(-z_0) * (1 - (z[k] - z_0) + (z[k] - z_0))
            sigma_upper = p2 * cvp.exp(-z_0) * (1 - (z[k] - z_0))

            constr += [gam[k] <= sigma_upper]
            constr += [gam[k] >= sigma_lower]
            constr += [z[k] >= z_0] 
            constr += [z[k] <= z_1]


    obj = cvp.norm(x[0:3, N-1] - q[:])
    problem = cvp.Problem(cvp.Minimize(obj), constr)
    problem.solve(solver=cvp.ECOS, verbose=True)

    return x.value, u.value, gam.value, z.value


if __name__ == "__main__":
    config = GFoldConfig(isp=350,
                         m=12000,
                         m_fuel=1000,
                         min_T_p=0.001*250000,
                         max_T_p=0.5*250000,
                         g=np.array([-3, 0, 0]),
                         pointing_lim=np.deg2rad(45),
                         landing_cone=np.deg2rad(10),
                         q=np.array([0, 0, 0]),
                         v_max=800,
                         w=np.array([0, 0, 0]),
                         start_pos=np.array([10000, 0, 0]),
                         start_vel=np.array([-300, 200, 200])
                         )

    x, u, gam, z = solve_gfold(config)
    print(f"final values:\nx: {x[:,-1]}\nu: {u[:,-1]}", )

    plt.plot(np.exp(z))
    plt.show()

    f = mlab.figure(bgcolor=(0, 0, 0))
    points3d([0], [0], [0], scale_factor=80.0, resolution=128, color=(0, 0.5, 0.5))
    s = plot3d(x[0,:], x[1,:], x[2,:], tube_radius=5.0, colormap='Spectral')
    v = quiver3d(x[0,:], x[1,:], x[2,:], u[0,:], u[1,:], u[2,:])
    mlab.show()