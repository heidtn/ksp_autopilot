import numpy as np
from mayavi import mlab
from mayavi.mlab import points3d, plot3d
import calculate_orbit
from tqdm import tqdm
import matplotlib.pyplot as plt


class BurnSolver:
    def __init__(self, max_time, orbit_config):
        self.max_time = max_time
        self.orbit_config = orbit_config
        self.orbit_simulator = calculate_orbit.SimulateOrbit(orbit_config)

    def forward_euler_with_escape(self, initial_vel, initial_pos, h=1):
        positions, velocities = self.orbit_simulator.run_simulation(initial_pos, initial_vel, h=1, time=self.max_time)
        return positions[-1], velocities[-1]

    def get_jacobian(self, cur_v, initial_pos):
        """
        Where cur_v is the current velocity to calculate around.  Uses finite differences.
        """
        J = np.zeros((3, 3))  # derivative of final position with respect to start velocity?
        # perturb the starting velocity in each direction just a little!
        eps = 1e-5
        for i in range(3):
            v = np.array(cur_v)  # make a copy
            v[i] += eps
            x_inc, _ = self.forward_euler_with_escape(v, initial_pos)
            v = np.array(cur_v)  # make another copy
            v[i] -= eps
            x_dec, _ = self.forward_euler_with_escape(v, initial_pos)
            J[:, i] = (x_inc - x_dec) / (2 * eps)
        
        return J

    def cost_func(self, final_pos, goal_pos):
        # TODO should this be a norm?
        return np.abs(goal_pos - final_pos)

    def shoot(self, initial_pos, initial_vel, goal_pos, iterations=300):
        error = []
        next_vel = initial_vel
        for i in tqdm(range(iterations)):
            #positions, velocities = self.orbit_simulator.run_simulation(initial_pos, next_vel, h=1, time=self.max_time)
            #self.orbit_simulator.plot_positions(positions)

            final_pos, final_vel = self.forward_euler_with_escape(next_vel, initial_pos)
            cost = self.cost_func(final_pos, goal_pos)
            error.append(np.linalg.norm(cost))
            jac = self.get_jacobian(next_vel, initial_pos)
            print("Jac: ", jac)
            print("Cost: ", np.linalg.norm(cost))
            next_vel = next_vel - 0.003 * np.dot(np.linalg.inv(jac), cost)
            
            print("Next vel test: ", next_vel)
        plt.plot(error)
        plt.show()

        return next_vel, cost, error


if __name__ == "__main__":
    pass


"""
Notes:
- given initial conditions, find a deltav that will land the ship at a particular position
- use nonlinear equations, include drag, r^2, etc.
- newton's method for shooting

This approach isn't optimal!  There are probably many global minima and this may not converge.  The 
starting guess for velocity is very important as is the max time to run to!
"""