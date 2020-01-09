import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import orbital_calculations

# Specific gas constant, this looks like what KSP uses for some reason?  Not the correct constant
R = 193.359

# TODO probably move these tables into a config file somwhere
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

EQUATOR_PASS_THRESHOLD = 1000.0
# TODO heidt, this is a pretty lazy way to do this, maybe change to time?
MAX_RUN_ITERATIONS = 100000
EQUATORIAL_LANDING_RANGE = 100000.0

#TODO eventually migrate to 3.7 so we can use dataclasses

class OrbitState:
    def __init__(self,
                 position: np.array,
                 velocity: np.array,
                 r_prev: float):
        self.position = position
        self.velocity = velocity
        self.r_prev = r_prev


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

    def calculate_aero(self, position, velocity, dt=0.1, max_orbits=10, plot=True):
        position = np.array(position)
        velocity = np.array(velocity)
        position_list = []
        time_list = []
        velocity_list = []

        run = True
        apoapsis = False
        periapsis = False
        num_orbits = 0
        iteration = 0

        print("position start: ", position)

        orbital_state = self.get_default_orbit_state(position, velocity)

        while run:
            orbital_state = self.step_orbit(orbital_state, dt)
            r = np.linalg.norm(orbital_state.position)
            r_prev = orbital_state.r_prev
            position_list.append(list(orbital_state.position))
            velocity_list.append(list(orbital_state.velocity))

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

            if num_orbits > max_orbits:
                run = False


            iteration += 1

        position_array = np.array(position_list)
        velocity_array = np.array(velocity_list)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(position_array[:, 0], position_array[:, 1], position_array[:, 2])
            self.plot_planet(ax)

            fig2 = plt.figure()
            plt.plot(np.linalg.norm(velocity_array, axis=1))

            plt.show()

        return position_array, velocity_array

    def get_default_orbit_state(self, position, velocity):
        position = np.array(position)
        velocity = np.array(velocity)
        return OrbitState(position, velocity, np.linalg.norm(position))

    def step_orbit(self, current_state, dt):
        # TODO is it better to unpack the state values and repack them at the end for readability?
        # Calc basic working values
        r = np.linalg.norm(current_state.position)
        g = self.mu / (r**2.0)
        altitude = r - self.R
        v = np.linalg.norm(current_state.velocity)

        # Get vectors based on implicit information
        inverse_normal_position = -current_state.position / np.linalg.norm(current_state.position)
        inverse_normal_velocity = -current_state.velocity / np.linalg.norm(current_state.velocity)
        gravity_vec = inverse_normal_position * g

        # Calculate the drag
        # Pressure from the surface not the center
        pressure = self.get_pressure(altitude)
        density = pressure / (R * self.get_atmospheric_temperature(altitude))
        Fdrag = self.Cd * self.area * 0.5 * density * v**2.0
        if altitude < self.H:
            Fdragvec = inverse_normal_velocity * Fdrag
        else:
            Fdragvec = np.array([0, 0, 0])

        # Calculate the next position
        acceleration = gravity_vec + Fdragvec / self.mass
        current_state.velocity += acceleration * dt
        current_state.position += current_state.velocity * dt

        current_state.r_prev = r
        return current_state

    def get_pressure(self, altitude):
        return np.interp(altitude, ATMOSPHERE_TABLE[:, 0], ATMOSPHERE_TABLE[:, 1])

    def get_atmospheric_temperature(self, altitude):
        return np.interp(altitude, TEMPERATURE_TABLE[:, 0], TEMPERATURE_TABLE[:, 1])

    def optimal_aero(self, position, velocity, apoapsis, periapsis, target_deltav, apsis_step=1000, apsis_start=370000):      
        position = np.array(position)
        velocity = np.array(velocity)  
        burn_found = False
        iteration = 0

        offset = periapsis - apsis_start
        while not burn_found:
            test_periapsis = periapsis - iteration * apsis_step - offset
            print("testing periapsis: ", test_periapsis, periapsis)
            new_test = self.generate_new_velocity(velocity, apoapsis, periapsis, test_periapsis)
            test_state = self.get_default_orbit_state(position, new_test)
            num_passes, success = self.test_burn_for_equator(test_state, target_deltav)
            print("Periapsis burn result: ", success)
            if success:
                return num_passes, test_periapsis

            if test_periapsis < self.R:
                return None, None

            iteration += 1

    def generate_new_velocity(self, velocity, apoapsis, periapsis, new_periapsis):
        """Generates a new velocity vector for a new periapsis when retrograde burning at the apoapsis"""
        print("calculating: ", apoapsis, periapsis, new_periapsis)
        deltav = orbital_calculations.apsis_change_dv(self.mu, apoapsis, periapsis, new_periapsis)
        print("testing dv: ", deltav)
        unit_vector_velocity = velocity / np.linalg.norm(velocity)
        new_velocity = velocity + unit_vector_velocity * deltav
        return new_velocity


    def test_burn_for_equator(self, orbital_state, target_deltav):
        """Returns True if the burn here uses less dv than our target"""

        positions, velocities = self.calculate_aero(orbital_state.position, orbital_state.velocity, dt=1.0)
        best_dv = -1.0
        best_position = None
        for i in range(len(positions)):
            required_deltav = self.horizontal_cancel_and_burn_dv(positions[i], velocities[i])

            if abs(positions[i, 2]) < EQUATORIAL_LANDING_RANGE:
                if required_deltav < best_dv or best_dv < 0:
                    best_dv = required_deltav
                if required_deltav < target_deltav:
                    best_position = positions[i, 2]

        print("best dv: ", best_dv, best_position)
        return 0, False


    def horizontal_cancel_and_burn_dv(self, position, velocity):
        """Calculates the dv for a horizontal cancel and suicide burn"""
        altitude = np.linalg.norm(position) - self.R
        # TODO This takes into account vertical velocity, figure out how to only use horizontal velocity
        # likely requires looking into reference frames a bit
        vel_mag = np.linalg.norm(velocity)
        g = self.mu / np.linalg.norm(position)**2.0
        # we're assuming constant gravity, this is mostly ok
        velocity_at_msl = np.sqrt(2.0*altitude*g)
        total_dv = vel_mag #+ velocity_at_msl
        return total_dv

    def plot_planet(self, ax):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*self.R
        y = np.sin(u)*np.sin(v)*self.R
        z = np.cos(v)*self.R
        ax.plot_wireframe(x, y, z, color="r")

    def run_to_equator(self, orbital_state, dt=0.5):
        """
        Forwards the orbital state until it passes over the equator.  Assumes it eventually does.
        Returns a tuple (orbital_state, status) where status is True if it successfully reached the equator
        """
        run = True
        iterations = 0

        position_list = []
        last_latitude = abs(orbital_state.position[2])
        approaching = False

        apoapsis = False
        periapsis = False
        r = np.linalg.norm(orbital_state.position)
        r_prev = r


        while run:
            r = np.linalg.norm(orbital_state.position)
            if not apoapsis and not periapsis:
                if r < r_prev:
                    apoapsis = True
                elif r > r_prev:
                    periapsis = True
            elif not apoapsis and r < r_prev:
                apoapsis = True
                periapsis = False
                print("Apoapsis at: ", r - self.R)
            elif not periapsis and r > r_prev:
                periapsis = True
                apoapsis = False
                print("Periapsis at: ", r - self.R)
            r_prev = r


            orbital_state = self.step_orbit(orbital_state, dt)
            position_list.append(np.array(orbital_state.position))
            if not approaching and abs(orbital_state.position[2]) < last_latitude:
                approaching = True

            if approaching and abs(orbital_state.position[2]) > last_latitude:
                print("Passing equator!")


                return orbital_state, True
            if iterations > MAX_RUN_ITERATIONS:
                return orbital_state, False
            if np.linalg.norm(orbital_state.position) < self.R:
                print("Crashed at: ", orbital_state.position, " Moving: ", np.linalg.norm(orbital_state.velocity))
                return orbital_state, False

            last_latitude = abs(orbital_state.position[2])
            iterations += 1
        
        return orbital_state, False


