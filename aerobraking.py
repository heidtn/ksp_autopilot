import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple

# Specific gas constant, this looks like what KSP uses for some reason?  Not the correct constant
R = 193.359

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

TEMPERATURE_TABLE = np.array([
    [0, 240],
    [5, 235],
    [10, 225],
    [20, 170],
    [25, 150],
    [45, 150],
    [50, 175]
])

OrbitState = namedtuple('OrbitState', ['position', 'velocity', 'r_prev'])


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

    def calculate_aero(self, position, velocity, plot=True):
        position = np.array(position)
        velocity = np.array(velocity)
        position_list = []
        time_list = []
        velocity_list = []
        density_list = []

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
            elif not apoapsis and r < r_prev:
                apoapsis = True
                periapsis = False
                print("Apoapsis at: ", r - self.R)
            elif not periapsis and r > r_prev:
                periapsis = True
                apoapsis = False
                print("Periapsis at: ", r - self.R)
                num_orbits += 1

            if r < self.R:
                print("Crashed at: ", position, " Moving: ", np.linalg.norm(velocity))
                run = False

            if num_orbits > 4:
                run = False

            t = iteration * dt
            g = self.mu / (r**2.0)
            altitude = r - self.R

            v = np.linalg.norm(velocity)
            velocity_list.append(v)
            inverse_normal_position = -position / np.linalg.norm(position)
            inverse_normal_velocity = -velocity / np.linalg.norm(velocity)
            gravity_vec = inverse_normal_position * g

            # Pressure from the surface not the center
            pressure = self.get_pressure(altitude)
            density = pressure / (R * self.get_atmospheric_temperature(altitude))
            density_list.append(density)

            Fdrag = self.Cd * self.area * 0.5 * density * v**2.0

            if altitude < self.H:
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

        fig2 = plt.figure()
        plt.plot(density_list)

        fig3 = plt.figure()
        plt.plot(velocity_list)

        plt.show()

    def step_orbit(self, current_state):
        

    def get_pressure(self, altitude):
        return np.interp(altitude, ATMOSPHERE_TABLE[:, 0], ATMOSPHERE_TABLE[:, 1])

    def get_atmospheric_temperature(self, altitude):
        return np.interp(altitude, TEMPERATURE_TABLE[:, 0], TEMPERATURE_TABLE[:, 1])

    def optimal_aero(self, position, velocity, apoapsis, periapsis, target_latitude, deltav):
        gradient_change = 1.0
        # test a new periapsis
        # get the deltav at the longitude
            # this deltav is the deltav for horizontal cancel + deltav for vertical burn
        # use the error between that deltav and the desired to adjust the periapsis
            # we want the highest periapsis that will yield below desired deltav
