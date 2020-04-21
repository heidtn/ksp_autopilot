import krpc
import clive_log
import calculate_orbit
import numpy as np
import matplotlib.pyplot as plt
import burn_solver

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

    def get_position_and_velocity(self):
        position = self.vessel.position(self.body.orbital_reference_frame)
        velocity = self.vessel.velocity(self.body.orbital_reference_frame)
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

    def do_landing_burn(self, end_velocity=0.0):
        self.wait_for_altitude(10000)
        self.point_retrograde()
        flight = self.vessel.flight()
        orbit = self.vessel.orbit
        starting_velocity = self.get_surface_speed()
        starting_altitude = flight.surface_altitude
        thrust_arr = []
        altitude_arr = []
        velocity_arr = []
        target_vels = []

        while self.vessel.situation != self.conn.space_center.VesselSituation.landed:
            current_altitude = flight.surface_altitude
            target_velocity = (1 - (1 - (current_altitude) / (starting_altitude))**2.0) * starting_velocity
            current_velocity = self.get_surface_speed()
            target_thrust = (current_velocity - target_velocity) * GAIN
            self.vessel.control.throttle = target_thrust
            thrust_arr.append(target_thrust)
            altitude_arr.append(current_altitude)
            velocity_arr.append(current_velocity)
            target_vels.append(target_velocity)
        self.vessel.control.throttle = 0.0

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time')
        ax1.set_ylabel('thrust fraction')
        p1 = ax1.plot(thrust_arr, label='Thrust')
        ax2 = ax1.twinx()
        ax2.set_ylabel('altitude (m)')
        p2 = ax2.plot(altitude_arr, '-r', label='Altitude')
        leg = p1 + p2
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("time")
        ax1.set_ylabel("velocity (m/s)")
        ax1.plot(velocity_arr, label='True Velocity')
        ax1.plot(target_vels, label='Target Velocity')
        ax1.legend()
        plt.show()

    def get_k(self):
        pass

    def solve_deltav_change(self, goal_pos, initial_guess):
        solver = burn_solver.BurnSolver(60*120, self.get_orbit_config())
        pos, vel = self.get_position_and_velocity()
        solver.shoot(pos, initial_guess, goal_pos) 

if __name__ == "__main__":
    autopilot = AutoPilot()
    autopilot.print_name()
    autopilot.print_position_and_velocity()
    #autopilot.plot_orbit()
    #autopilot.do_landing_burn()
    goal_pos = np.array((174214.6033875, -251353.7457310537, 95648.02985248323))
    pos, vel = autopilot.get_position_and_velocity()
    guess = np.array(vel) * 0.5
    autopilot.solve_deltav_change(goal_pos, guess)
