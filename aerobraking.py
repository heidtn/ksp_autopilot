import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Specific gas constant
R = 287.053

ATMOSPHERE_TABLE = np.array([
    [0	    ,6755],
    [2500	,5136],
    [5000	,3774],
    [7500	,2663],
    [10000	,1797],
    [15000	,716.9],
    [20000	,241.0],
    [25000	,91.73],
    [30000	,32.90],
    [40000	,4.918],
    [50000	,0     ]
])


class AerobrakeCalculator:
    def __init__(self, Cd, mu, R, A, mass, H, T_0):
        """
        Cd:   coefficient of drag
        mu:   gravity coefficient
        R     radius of the planet
        Bc:   Ballistic coefficient
        mass: mass of the spacecraft
        H:    atmospheric height
        P_0:  pressure at 0 altitude
        T_0:  temperature at sea level
        """
        self.Cd = Cd
        self.mu = mu
        self.R = R
        self.area = A
        self.mass = mass
        self.H = H
        #self.area = mass / (Bc  * Cd)

        print("Area: ", self.area)

        self.T_0 = T_0
        # TODO: calculate the temperature for various altitudes/positions/etc.
        self.get_density = lambda p: p / (R * T_0)
        self.get_drag = lambda v, p: Cd * self.area * 0.5 * p * v**2.0

    def calculate_aero(self, position, velocity, plot=True):
        position = np.array(position)
        velocity = np.array(velocity)
        position_list = []
        time_list = []

        r = np.linalg.norm(position)
        r_prev = np.linalg.norm(position)

        run = True
        apoapsis = False
        periapsis = False
        num_orbits = 0
        iteration = 0
        dt = 0.1

        print("position start: ", position)

        while run:
            r = np.linalg.norm(position)

            if not apoapsis and not periapsis:
                if r < r_prev:
                    apoapsis = True
                else:
                    periapsis = True

            if not apoapsis and r < r_prev:
                apoapsis = True
                periapsis = False
                print("Apoapsis at: ", r - self.R)

            if not periapsis and r > r_prev:
                periapsis = True
                apoapsis = False
                print("Periapsis at: ", r - self.R)
                num_orbits += 1

            if r < self.R:
                print("Crashed at: ", position)
                run = False

            if num_orbits > 10:
                run = False

            if iteration % 100000 == 0:
                print(iteration)

            t = iteration * dt
            g = self.mu / (r**2.0)

            v = np.linalg.norm(velocity)
            inverse_normal_position = -position / np.linalg.norm(position)
            inverse_normal_velocity = -velocity / np.linalg.norm(velocity)
            gravity_vec = inverse_normal_position * g

            # Pressure from the surface not the center
            pressure = self.get_pressure(r - self.R)
            density = self.get_density(pressure)
            Fdrag = self.get_drag(v, density)
            if r - self.R < self.H:
                Fdragvec = inverse_normal_velocity * Fdrag
            else:
                Fdragvec = np.array([0, 0, 0])

            acceleration = gravity_vec + Fdragvec / self.mass
            velocity += acceleration * dt
            position += velocity * dt

            position_list.append(np.array(position))
            time_list.append(t)

            r_prev = r
            iteration += 1


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        position_array = np.array(position_list)
        ax.plot(position_array[:, 0], position_array[:, 1], position_array[:, 2])
        plt.show()

    def get_pressure(self, altitude):
        return np.interp(altitude, ATMOSPHERE_TABLE[:, 0], ATMOSPHERE_TABLE[:, 1])

