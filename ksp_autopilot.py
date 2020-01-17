import krpc
import clive_log
import calculate_orbit


class AutoPilot:
    def __init__(self):
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.body = self.vessel.orbit.body

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
        mu = self.body.gravitational_parameter
        radius = self.body.equatorial_radius
        orbit_simulator = calculate_orbit.SimulateOrbit(mu, radius)
        positions, velocities = orbit_simulator.run_simulation(p, v)
        orbit_simulator.plot_positions(positions)


if __name__ == "__main__":
    autopilot = AutoPilot()
    autopilot.print_name()
    autopilot.print_position_and_velocity()
    autopilot.plot_orbit()