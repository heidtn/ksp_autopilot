import numpy as np
import matplotlib as plt
from control.matlab import lqr
from collections import namedtuple

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt  

m = GEKKO()

ControlConfig = namedtuple('ControlConfig', ['drag_coeffs', 'drag_linearization_v', 'mass', 'Q', 'R'])
MPCConfig = namedtuple('MPCConfig', ['MOIs', 'max_thrust', 'fuel_depletion_rate', 'g'])

def create_K(config):
    g = config.g
    drag_at_point = config.drag_coeffs * drag_linearization_v**2.0
    drag_factor = (drag_at_point/drag_linearization_v) / config.mass
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, -drag_factor, 0, 0],
                  [0, 0, 0, 0, -drag_factor, 0],
                  [0, 0, 0, 0, 0, -drag_factor]])
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3)

    K, X, E = lqr(A, B, config.Q, config.R)
    return np.array(K)


class RocketMPC:
    def __init__(self, mpc_config):
        self.config = mpc_config

    def build(self):
        self.m = GEKKO(remote=False)
        m = self.m
        m.time = np.linspace(0,5,20)

        Ia = self.config.MOIs[0]
        Ib = self.config.MOIs[1]

        # Parameters
        self.mass = m.Param()

        # Manipulated variable
        self.T = m.MV(value=0, lb=0, ub=self.config.max_thrust)
        T = self.T
        T.STATUS = 1  # allow optimizer to change
        T.DCOST = 0.1 # smooth out gas pedal movement
        T.DMAX = 20   # slow down change of gas pedal

        # Manipulated variable
        self.Ta = m.MV(value=0, lb=-1.0, ub=1.0)
        Ta = self.Ta
        Ta.STATUS = 1  # allow optimizer to change
        Ta.DCOST = 0.1 # smooth out gas pedal movement
        Ta.DMAX = 20   # slow down change of gas pedal

        # Manipulated variable
        self.Tb = m.MV(value=0, lb=-1.0, ub=1.0)
        Tb = self.Tb
        Tb.STATUS = 1  # allow optimizer to change
        Tb.DCOST = 0.1 # smooth out gas pedal movement
        Tb.DMAX = 20   # slow down change of gas pedal

        # Controlled Variable
        self.X = m.Array(m.CV, (3))
        X = self.X
        for x in X:
            x.STATUS = 1  # add the SP to the objective
            m.options.CV_TYPE = 2 # squared error
            x.TR_INIT = 1 # set point trajectory
            x.TAU = 5     # time constant of trajectory
        self.X[0].LOWER = 0.0

        self.X_dot = m.Array(m.CV, (3))
        X_dot = self.X_dot
        for x in X_dot:
            x.STATUS = 1  # add the SP to the objective
            m.options.CV_TYPE = 2 # squared error
            x.TR_INIT = 1 # set point trajectory
            x.TAU = 5     # time constant of trajectory

        self.a = m.CV(0, lb=-np.pi/2, ub=np.pi/2)
        a = self.a
        a_dot = m.CV(0)
        self.b = m.CV(0, lb=-np.pi/2, ub=np.pi/2)
        b = self.b
        b_dot = m.CV(0)

        m.Equation(X[0].dt() == X_dot[0])
        m.Equation(X[1].dt() == X_dot[1])
        m.Equation(X[2].dt() == X_dot[2])

        m.Equation(X_dot[0].dt() == m.cos(a)*m.cos(b)*T - self.config.g)
        m.Equation(X_dot[1].dt() == m.sin(a)*T)
        m.Equation(X_dot[2].dt() == m.cos(a)*m.sin(-b)*T)
        
        m.Equation(a.dt() == a_dot)
        m.Equation(a_dot.dt() == Ta * Ia)

        m.Equation(b.dt() == b_dot)
        m.Equation(b_dot.dt() == Tb * Ib)

        m.options.IMODE = 6 # control
        
    def solve_next_state(self, current_state, goal_state):
        self.X[0].SP = goal_state[0]
        self.X[1].SP = goal_state[1]
        self.X[2].SP = goal_state[2]

        self.X_dot[0].SP = goal_state[3]
        self.X_dot[1].SP = goal_state[4]
        self.X_dot[2].SP = goal_state[5]

        self.X[0].value = current_state[0]
        self.X[1].value = current_state[1]
        self.X[2].value = current_state[2]

        self.X_dot[0].value = current_state[3]
        self.X_dot[1].value = current_state[4]
        self.X_dot[2].value = current_state[5]

        self.m.solve(disp=False)
        return self.X, self.X_dot, self.T, self.Ta, self.Tb, self.a, self.b

    def get_next_control(self, current_state, goal_state):
        x, xd, T, Ta, Tb, a, b = self.solve_next_state(current_state, goal_state)
        return T.value[0], Ta.value[0], Tb.value[0]


if __name__ == "__main__":
    config = MPCConfig(MOIs=np.array([10, 10, 10]), max_thrust=100, fuel_depletion_rate=0, g=3.2)
    mpc = RocketMPC(config)
    print("Building mpc")
    mpc.build()
    print("executing")
    import time
    t1 = time.time()
    for i in range(10):
        current_state = np.array([10, 0, 10, 1, 0, -10])
        goal_state = np.array([0, 0, 0, 0, 0, 0])
        X, X_dot, T, Ta, Tb, a, b = mpc.solve_next_state(current_state, goal_state)
        print(X[2].value)
    t2 = time.time()
    print("Elapsed: ", t2 - t1)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(list(T.value))
    plt.figure()
    plt.plot(Ta.value)
    plt.show()