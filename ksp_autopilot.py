import krpc
import clive_log
import calculate_orbit
import numpy as np
import matplotlib.pyplot as plt
import burn_solver
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import gfold_test

from mayavi import mlab
from mayavi.mlab import points3d, plot3d, quiver3d

CROSS_SECTION_AREA = 5.381
COEFFICIENT_OF_DRAG = 1.455
GAIN = 0.05


class AutoPilot:
    def __init__(self):
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.body = self.vessel.orbit.body
        self.autopilot = self.vessel.auto_pilot

    def print_name(self):
        name = self.vessel.name
        print(f"the vessel's name is: {name}")

    def print_position_and_velocity(self):
        p, v = self.get_position_and_velocity()
        print(f"Position: {p}\nVelocity: {v}")

    def get_position_and_velocity(self, frame=None):
        if not frame:
            frame = self.body.orbital_reference_frame
        position = self.vessel.position(frame)
        velocity = self.vessel.velocity(frame)
        return position, velocity

    def stream_position_and_velocity(self):
        context = clive_log.Context("vpstream")
        context.add_text_field("velocity")
        context.add_text_field("position")
        while True:
            p, v = self.get_position_and_velocity()
            context.write_text_field("position", f"Position: {p}")
            context.write_text_field("velocity", f"Velocity: {v}")
            context.display()

    def plot_orbit(self):
        p, v = self.get_position_and_velocity()
        config = self.get_orbit_config()
        orbit_simulator = calculate_orbit.SimulateOrbit(config)
        positions, velocities = orbit_simulator.run_simulation(p, v)
        orbit_simulator.plot_positions(positions)

    def get_orbit_config(self):
        mu = self.body.gravitational_parameter
        radius = self.body.equatorial_radius
        cross_section_area = CROSS_SECTION_AREA
        drag_coefficient = COEFFICIENT_OF_DRAG
        ship_mass = self.vessel.mass
        ref_frame = self.body.orbital_reference_frame
        density_function = lambda x: self.body.atmospheric_density_at_position(x, ref_frame)
        config = calculate_orbit.OrbitConfig(mu=mu, planet_radius=radius, cross_section_area=cross_section_area,
                                             drag_coefficient=drag_coefficient, density_function=density_function,
                                             ship_mass=ship_mass)
        return config

    def get_surface_speed(self):
        ref_frame = self.conn.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame) 
        velocity = self.vessel.flight(ref_frame).velocity
        return np.linalg.norm(velocity)

    def get_burn_altitude(self, thrust_percentage=1.0):
        max_thrust = self.vessel.max_thrust * thrust_percentage
        surface_grav = self.vessel.orbit.body.surface_gravity
        cur_vel = self.get_surface_speed()
        cur_pos = self.vessel.flight().surface_altitude
        distance = cur_vel**2.0 * self.vessel.mass / (2*(max_thrust - surface_grav*self.vessel.mass))

        return distance

    def point_retrograde(self):
        self.autopilot.sas = True
        self.autopilot.sas_mode = self.conn.space_center.SASMode.retrograde
        self.autopilot.wait()

    def wait_for_altitude(self, altitude):
        flight = self.vessel.flight()
        while flight.surface_altitude > altitude:
            pass
    
    def create_gfold_config(self):
        config = gfold_test.GFoldConfig(isp=self.vessel.vacuum_specific_impulse,
                            m=self.vessel.mass,
                            m_fuel=self.vessel.mass - self.vessel.dry_mass,
                            p1=0.0,#01*self.vessel.max_thrust,
                            p2=0.5*self.vessel.max_thrust,
                            g=np.array([-self.body.surface_gravity, 0, 0]),
                            pointing_lim=np.deg2rad(65),
                            landing_cone=np.deg2rad(10),
                            q=np.array([0, 0, 0]),
                            v_max=1000,
                            w=np.array([0, 0, 0])
                            )
        print("Config: ", config)
        return config

    def do_landing_burn(self, goal_position):
        #self.wait_for_altitude(10000)
        #self.point_retrograde()
        flight = self.vessel.flight()
        orbit = self.vessel.orbit
        # create a transform for the goal position as an x-up frame
        # TODO: This is lazy and will fail if we are landing at the poles
        x_ax = goal_position / np.linalg.norm(goal_position)
        y_ax = np.cross(x_ax, np.array([0, 0, 1]))
        z_ax = np.cross(x_ax, y_ax)
        y_ax /= np.linalg.norm(y_ax)
        z_ax /= np.linalg.norm(z_ax)
        rot = np.column_stack([x_ax, y_ax, z_ax])
        q = R.from_matrix(rot).as_quat()


        ref_frame = self.conn.space_center.ReferenceFrame.create_relative(self.body.reference_frame,
                                                          position=goal_position,
                                                          rotation=q)


        pos, vel = self.get_position_and_velocity(ref_frame)
        print("POS VEL: ", pos, vel)
        start_time = self.conn.space_center.ut
        config = self.create_gfold_config()
        x, u, gam, z, dt = gfold_test.solve_gfold(config, pos, vel)


        f = mlab.figure(bgcolor=(0, 0, 0))
        points3d([0], [0], [0], scale_factor=200.0, resolution=128, color=(0, 0.5, 0.5))
        s = plot3d(x[0,:], x[1,:], x[2,:], tube_radius=5.0, colormap='Spectral')
        v = quiver3d(x[0,:], x[1,:], x[2,:], u[0,:], u[1,:], u[2,:])
        mlab.axes()
        mlab.show()

        timesteps = np.linspace(0, x.shape[1]*dt, num=x.shape[1])
        f = interp1d(timesteps, x[:], kind='cubic')
        u = interp1d(timesteps, u, kind='cubic')
        gam = interp1d(timesteps, gam, kind='cubic')

        self.autopilot.reference_frame = ref_frame
        self.autopilot.engage()

        while self.vessel.situation != self.conn.space_center.VesselSituation.landed:
            cur_time = self.conn.space_center.ut - start_time
            index_time = cur_time
            try:
                goal_position = f(index_time)
            except ValueError:
                goal_position = f(x.shape[1]*dt)
            pos, vel = self.get_position_and_velocity(ref_frame)
            err = goal_position - np.array([*pos, *vel])

            thrust_vector = np.array([2., 0, 0])
            thrust_vector[1:3] = err[1:3]*0.025
            thrust_vector[1:3] += err[4:6]*0.56
            self.autopilot.target_direction = tuple(thrust_vector)

            print("Thrust: ", thrust_vector)
            print("Err: ", err)
            thrust_vector[0] = 0.0
            thrust_vector *= 0.05
            thrust_vector[0] = err[0]*2.0
            thrust_vector[0] += err[3]*4.3
            thrust_vector[0] = np.max((thrust_vector[0], 0))
            throttle = np.linalg.norm(thrust_vector)*0.08   #0.25
            print("Thrustvec2: ", throttle)
            self.vessel.control.throttle  = throttle


        self.vessel.control.throttle = 0


    def solve_deltav_change(self, goal_pos, initial_guess, time=60*240):
        solver = burn_solver.BurnSolver(time, self.get_orbit_config())
        pos, vel = self.get_position_and_velocity()
        solver.shoot3d(pos, initial_guess, goal_pos) 

if __name__ == "__main__":
    autopilot = AutoPilot()
    autopilot.print_name()
    autopilot.print_position_and_velocity()
    pos, vel = autopilot.get_position_and_velocity(autopilot.body.reference_frame)
    print("Rotating frame position: ", pos)

    #goal_pos = np.array((159198.49890755463, -1176.8349779020318, -578566.3171298194))
    goal_pos = np.array((-160822.82145683363, -130782.2978001064, -249393.70446923468))
    autopilot.do_landing_burn(goal_pos)
