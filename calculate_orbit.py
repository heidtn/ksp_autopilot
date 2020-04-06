import numpy as np
from mayavi import mlab
from mayavi.mlab import points3d, plot3d
from collections import namedtuple

OrbitConfig = namedtuple("OrbitConfig", ["mu",
                                         "planet_radius",
                                         "cross_section_area",
                                         "drag_coefficient",
                                         "density_function",
                                         "ship_mass"])


class SimulateOrbit:
    def __init__(self, config):
        self.mu = config.mu
        self.planet_radius = config.planet_radius
        self.cross_section_area = config.cross_section_area
        self.drag_coefficient = config.drag_coefficient
        self.density_function = config.density_function
        self.ship_mass = config.ship_mass

    def run_simulation(self, start_position, start_velocity, dt=1, iterations=100000):
        position = np.array(start_position)
        velocity = np.array(start_velocity)
        positions = []
        velocities = []

        for i in range(iterations):
            acceleration = self.get_gravity(position) + self.get_drag(position, velocity)
            velocity += acceleration * dt
            position += velocity * dt
            positions.append(np.array(position))
            velocities.append(np.array(velocity))
            if np.linalg.norm(position) < self.planet_radius:
                break
        
        return np.array(positions), np.array(velocities)

    def get_drag(self, position, velocity):
        v_mag = np.linalg.norm(velocity)
        density = self.density_function(position)
        drag_force = 0.5 * density * self.drag_coefficient * self.cross_section_area * v_mag**2.0
        drag_decelleration_mag = drag_force / self.ship_mass
        v_unit_vec = velocity / v_mag
        drag_decelleration_vec = -v_unit_vec * drag_decelleration_mag
        return drag_decelleration_vec

    def get_gravity(self, position):
        distance_from_center = np.linalg.norm(position)
        gravitational_acceleration = self.mu / distance_from_center**2.0
        unit_position = position / distance_from_center
        gravity_vector = -unit_position * gravitational_acceleration
        return gravity_vector

    def plot_positions(self, position_array, animate=False):
        f = mlab.figure(bgcolor=(0, 0, 0))
        points3d([0], [0], [0], scale_factor=self.planet_radius*2.0, resolution=128, color=(0, 0.5, 0.5))
        tube_size = self.planet_radius / 100.0
        points3d(position_array[0, 0], position_array[0, 1], position_array[0, 2],
                 scale_factor=tube_size*10, resolution=128, color=(0.5, 0, 0))

        s = plot3d(position_array[:, 0], position_array[:, 1], position_array[:, 2],
                   tube_radius=tube_size, colormap='Spectral')

        @mlab.animate(delay=10)
        def anim():
            f = mlab.gcf()
            while True:
                for i in range(1, len(position_array) // 50):
                    s.mlab_source.reset(x=position_array[:i*50, 0], y=position_array[:i*50, 1], z=position_array[:i*50, 2])
                    yield
        if animate:
            anim()
        mlab.show()

if __name__ == "__main__":
    earth_mu = 3.986004418e14
    earth_radius = 6.371e6
    iss_altitude = 408000.0
    iss_speed = 7660.0
    start_position = np.array([iss_altitude + earth_radius, 0, 0])
    start_velocity = np.array([0, iss_speed, 0])
    orbit_simulator = SimulateOrbit(earth_mu, earth_radius)
    positions, velocities = orbit_simulator.run_simulation(start_position, start_velocity)
    orbit_simulator.plot_positions(positions)