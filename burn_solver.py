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

    def forward_euler_with_escape(self, initial_vel, initial_pos, h=10):
        positions, velocities = self.orbit_simulator.run_simulation(initial_pos, initial_vel, h=h, time=self.max_time)
        #self.orbit_simulator.plot_positions(positions)
        return positions[-1], velocities[-1]

    def get_diff(self, initial_vel_dir, initial_pos, goal_pos, vel_mag):
        """
        Where cur_v is the current velocity to calculate around.  Uses finite differences.
        """
        # perturb the starting velocity in each direction just a little!
        eps = 1e-6
        v = initial_vel_dir * (vel_mag + eps)  # make a copy
        x_inc, _ = self.forward_euler_with_escape(v, initial_pos)
        cost_inc = self.cost_func(x_inc, goal_pos)

        v = initial_vel_dir * (vel_mag - eps)
        x_dec, _ = self.forward_euler_with_escape(v, initial_pos)
        cost_dec = self.cost_func(x_dec, goal_pos)
        
        J = (cost_inc - cost_dec) / (2 * eps)
        
        return J

    def get_jacobian(self, cur_v, initial_pos, goal_pos):
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
            cost_inc = self.cost_func3d(x_inc, goal_pos)

            v = np.array(cur_v)  # make another copy
            v[i] -= eps
            x_dec, _ = self.forward_euler_with_escape(v, initial_pos)
            cost_dec = self.cost_func3d(x_dec, goal_pos)
            J[:, i] = (cost_inc - cost_dec) / (2 * eps)
        
        return J

    def cost_func(self, final_pos, goal_pos):
        return np.linalg.norm(goal_pos - final_pos)
    
    def cost_func3d(self, final_pos, goal_pos):
        return goal_pos - final_pos

    def conjugate_grad_Sn(self, cur_J, prev_J, prev_Sn):
        beta = (cur_J.T@(cur_J - prev_J))/(prev_J.T@prev_J)
        Sn = cur_J + beta*prev_Sn
        return Sn

    def shoot3d(self, initial_pos, initial_vel, goal_pos, iterations=85, alpha=0.1):
        error = []
        distance = []
        next_vel = initial_vel
        cur_Sn = None
        prev_J = None

        print("Got : ", initial_pos, initial_vel, goal_pos)
        for i in tqdm(range(iterations)):

            final_pos, final_vel = self.forward_euler_with_escape(next_vel, initial_pos)
            cost = self.cost_func3d(final_pos, goal_pos)
            error.append(np.linalg.norm(cost))
            distance.append(np.linalg.norm(final_pos - goal_pos))

            jac = self.get_jacobian(next_vel, initial_pos, goal_pos)
            cost = self.cost_func3d(final_pos, goal_pos)
            cur_grad = np.linalg.pinv(jac)@cost
            if cur_Sn is None:
                cur_Sn = cur_grad
            else:
                cur_Sn = self.conjugate_grad_Sn(cur_grad, prev_J, cur_Sn)
            next_vel = next_vel - alpha * cur_Sn

            prev_J = cur_grad
            plt.clf()
            plt.plot(distance)
            plt.pause(0.05)
            
            print("Next settings: ", next_vel, cost, error)
        plt.show()

        positions, velocities = self.orbit_simulator.run_simulation(initial_pos, next_vel, h=1, time=self.max_time)
        self.orbit_simulator.plot_positions(positions)
        print("Impact velocity: ", velocities[-1])

        return next_vel, cost, error

    def shoot(self, initial_pos, initial_vel, goal_pos, alpha=0.1, iterations=50):
        error = []
        distance = []
        initial_vel_dir = initial_vel / np.linalg.norm(initial_vel)
        next_vel_mag = np.linalg.norm(initial_vel)

        for i in tqdm(range(iterations)):
            next_vel = initial_vel_dir * next_vel_mag
            final_pos, final_vel = self.forward_euler_with_escape(next_vel, initial_pos)
            cost = self.cost_func(final_pos, goal_pos)
            error.append(np.linalg.norm(cost))
            distance.append(np.linalg.norm(final_pos - goal_pos))

            diff = self.get_diff(initial_vel_dir, initial_pos, goal_pos, next_vel_mag)
            print("distance: ", distance[-1])
            next_vel_mag = next_vel_mag - alpha * (cost / diff)

            plt.clf()
            plt.title("Error per iteration")
            plt.xlabel("iteration")
            plt.ylabel("error (m)")
            plt.plot(distance)
            plt.pause(0.05)
            
            print("Next vel test: ", next_vel_mag)
        #plt.plot(error)
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