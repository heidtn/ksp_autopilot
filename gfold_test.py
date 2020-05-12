import cvxpy as cvp
import numpy as np
from collections import namedtuple

from mayavi import mlab
from mayavi.mlab import points3d, plot3d, quiver3d
import matplotlib.pyplot as plt

"""
http://www.larsblackmore.com/iee_tcst13.pdf

carrying the same assumption here that X is up (i.e. normal to land)

TODO: 
    - fix q definition to be the y,z coordinate instead of xyz
    - figure out how to best pick an initial tf
    - make alpha a parameter istead of isp

"""


GFoldConfig = namedtuple('GFoldConfig', ['isp', 'm', 'm_fuel',
                                         'p1', 'p2',
                                         'g', 'pointing_lim', 'landing_cone',
                                         'q', 'v_max', 'w'])


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

class GFOLDSolverPosition:
    def __init__(self, config, iterations):
        N = iterations
        self.N = N
        self.config = config

        # Parameters
        self.dt = cvp.Parameter()
        self.x_0 = cvp.Parameter(6)

        # Solved variables
        self.x = cvp.Variable((6, N))
        self.u = cvp.Variable((3, N)) 
        self.gam = cvp.Variable(N)
        self.z = cvp.Variable(N)
        
        # Problem 3
        self.constr = []
        self.constr = set_initial_constraints(self.constr, config, self.x, self.u, self.gam, self.z, N, self.x_0)
        self.constr = running_constraints(self.constr, config, self.x, self.u, self.gam, self.z, self.dt, N)
        self.obj = cvp.norm(self.x[0:3, N-1] - config.q[:])
        self.problem = cvp.Problem(cvp.Minimize(self.obj), self.constr)

    def initialize_dt(self, start_vel):
        alpha = 1.0 / (config.isp * 9.8)
        t_max = self.config.m / (alpha * self.config.p2)  # this differs from the paper, but using p1 results in cvp.log to return nan
        t_min = (self.config.m - self.config.m_fuel) * np.linalg.norm(start_vel) / self.config.p2
        self.dt.value = (t_max + t_min) / 2.0 / self.N

    def solve(self, start_pos, start_vel):
        self.initialize_dt(start_vel)
        self.x_0.value = np.array([*start_pos, *start_vel])

        self.problem.solve(solver=cvp.ECOS, verbose=True)
        return self.x.value, self.u.value, self.gam.value, self.z.value
        

class GFOLDSolverMass(GFOLDSolverPosition):
    def __init__(self, config, iterations):
        super().__init__(config, iterations)
        self.d_p3 = cvp.Parameter(3)
        self.constr += [cvp.norm(self.x[:3, self.N-1] - config.q) <= cvp.norm(self.d_p3 - config.q)]
        self.problem = cvp.Problem(cvp.Maximize(self.z[self.N-1]), self.constr)

    def solve(self, d_p3, start_pos, start_vel):
        self.initialize_dt(start_vel)
        self.d_p3.value = d_p3
        self.x_0.value = np.array([*start_pos, *start_vel])

        self.problem.solve(solver=cvp.ECOS, verbose=True, max_iters=200)
        first_mass = self.z[-1]
        self.dt.value += 1e-3
        return self.x.value, self.u.value, self.gam.value, self.z.value


class GFOLDSolver:
    def __init__(self, config, iterations):
        self.config = config
        self.position_solver = GFOLDSolverPosition(config, iterations)
        self.mass_solver  = GFOLDSolverMass(config, iterations)

    def solve(self, start_pos, start_vel):
        print("Solving position problem")
        x1, u1, gam1, z1 = self.position_solver.solve(start_pos, start_vel)
        print("Solving mass problem")
        x2, u2, gam2, z2 = self.mass_solver.solve(x1[0:3,-1], start_pos, start_vel)
        return x2, u2, gam2, z2



def set_initial_constraints(constr, config, x, u, gam, z, N, x_0):
    #x_0 = np.array([*config.start_pos, *config.start_vel])

    constr += [x[:, 0] == x_0[:]]   # Initial velocity and position
    constr += [x[3:6, N-1] == np.array([0, 0, 0])]  # Final velocity == 0

    # TODO (make initial thrust direction a parameter)
    constr += [u[:, 0] == gam[0]*_e1] # Initial thrust is vertical
    constr += [u[:, N-1] == gam[N-1]*_e1] # final thrust is vertical (or 0)
    constr += [z[0] == cvp.log(config.m)] # Initial mass

    constr += [x[0, N-1] == 0]  # Final altitude should be 0
    constr += [x[0, 0:N-1] >= 0]  # All altitudes during flight should be above the ground
    return constr

def running_constraints(constr, config, x, u, gam, z, dt, N):
    A_w = create_A(config.w)
    alpha = 1.0 / (config.isp * 9.8)
    pointing_angle = np.cos(config.pointing_lim)
    p1 = config.p1
    p2 = config.p2
    v_max = config.v_max
    c = get_cone(config.landing_cone)
    g = config.g

    # Simple Euler integration
    for k in range(N-1):
        # Rocket dynamics constraints
        constr += [x[0:3, k+1] == x[0:3, k] + dt*(A_w@x[:, k])[0:3]]
        constr += [x[3:6, k+1] == x[3:6, k] + dt*(g + u[:, k])]
        constr += [z[k+1] == z[k] - dt*alpha*gam[k]]

        constr += [cvp.norm(x[3:6, k]) <= v_max]  # Velocity remains below maximum
        constr += [cvp.norm(u[:,k]) <= gam[k]]  # Helps enforce the magnitude of thrust vector == thrust magnitude
        constr += [u[0,k] >= pointing_angle*gam[k]]  # Rocket can only point away from vertical by so much
        constr += [cvp.norm(_E@(x[:3,k] - x[:3,-1])) - c.T@(x[:3, k] - x[:3,-1]) <= 0]  # Stay inside the glide cone

        if k > 0:
            z_0 = cvp.log(config.m - alpha * p2 * (k) * dt)
            z_1 = cvp.log(config.m - alpha * p1 * (k) * dt)

            sigma_lower = p1 * cvp.exp(-z_0) * (1 - (z[k] - z_0) + (z[k] - z_0))
            sigma_upper = p2 * cvp.exp(-z_0) * (1 - (z[k] - z_0))

            # Minimimum and maximum thrust constraints
            constr += [gam[k] <= sigma_upper]
            constr += [gam[k] >= sigma_lower]
            # Minimum and maximum mass constraints
            constr += [z[k] >= z_0] 
            constr += [z[k] <= z_1]
    return constr

def solve_gfold(config, start_pos, start_vel, iterations=100):
    solver = GFOLDSolver(config, iterations)
    x, u, gam, z = solver.solve(start_pos, start_vel)
    return x, u, gam, z


if __name__ == "__main__":
    config = GFoldConfig(isp=350,
                         m=12000,
                         m_fuel=1000,
                         p1=0.001*250000,
                         p2=0.5*250000,
                         g=np.array([-3, 0, 0]),
                         pointing_lim=np.deg2rad(45),
                         landing_cone=np.deg2rad(30),
                         q=np.array([0, 0, 0]),
                         v_max=1000,
                         w=np.array([0, 0, 0])
                         )

    start_pos = np.array([25000, 10000, 10000])
    start_vel = np.array([-300, -300, 300])

    x, u, gam, z = solve_gfold(config, start_pos, start_vel)
    print(f"final values:\nx: {x[:,-1]}\nu: {u[:,-1]}", )


    f = mlab.figure(bgcolor=(0, 0, 0))
    points3d([0], [0], [0], scale_factor=200.0, resolution=128, color=(0, 0.5, 0.5))
    s = plot3d(x[0,:], x[1,:], x[2,:], tube_radius=5.0, colormap='Spectral')
    v = quiver3d(x[0,:], x[1,:], x[2,:], u[0,:], u[1,:], u[2,:])
    mlab.axes()
    mlab.show()