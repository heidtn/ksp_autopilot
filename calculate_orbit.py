import numpy as np
from mayavi import mlab
from mayavi.mlab import points3d, plot3d

class SimulateOrbit:
    def __init__(self, mu, planet_radius):
        self.mu = mu
        self.planet_radius = planet_radius

    def run_simulation(self, start_position, start_velocity, dt=1, iterations=100000):
        position = np.array(start_position)
        velocity = np.array(start_velocity)
        positions = []
        velocities = []

        for i in range(iterations):
            acceleration = self.get_gravity(position)
            velocity += acceleration * dt
            position += velocity * dt
            positions.append(np.array(position))
            velocities.append(np.array(velocity))
        
        return np.array(positions), np.array(velocities)

    def get_gravity(self, position):
        distance_from_center = np.linalg.norm(position)
        gravitational_acceleration = self.mu / distance_from_center**2.0
        unit_position = position / distance_from_center
        gravity_vector = -unit_position * gravitational_acceleration
        return gravity_vector

    def plot_positions(self, position_array, animate=True):
        points3d([0], [0], [0], scale_factor=self.planet_radius*2.0, resolution=128, color=(0, 0.5, 0.5))
        s = plot3d(position_array[:, 0], position_array[:, 1], position_array[:, 2], tube_radius=50000.0, colormap='Spectral')

        @mlab.animate(delay=10)
        def anim():
            f = mlab.gcf()
            while True:
                for i in range(1, len(position_array) / 10):
                    s.mlab_source.reset(x=position_array[:i*10, 0], y=position_array[:i*10, 1], z=position_array[:i*10, 2])
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