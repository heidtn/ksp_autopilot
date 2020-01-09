import krpc
import aerobraking
import orbital_calculations
import numpy as np
import time

MIN_HORIZONTAL_SPEED = 10.0
MAX_THROTTLE = 1.0
THROTTLE_OFF = 0.0
LANDING_SPEED = 6.0
BREAKING_THREHOLD = 20.0
VESSEL_SURFACE_AREA = 5.381
DRAG_COEFF = 1.5
BREAKING_THRESHOLD = 20.0
MIN_BURN_ALTITUDE = 3000


class AutoPilot:
    def __init__(self):
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.flight = self.vessel.flight()
        self.autopilot = self.vessel.auto_pilot
        self.orbit = self.vessel.orbit

    def reinit_connection(self):
        self.conn = krpc.connect()

    def horizontal_burn(self):
        horizontal = self.get_horizontal_ground_vector()
        self.autopilot.target_direction = horizontal
        self.autopilot.engage()
        self.autopilot.wait()

        self.vessel.control.throttle = MAX_THROTTLE

        while self.vessel.flight(self.vessel.orbit.body.orbital_reference_frame).horizontal_speed > MIN_HORIZONTAL_SPEED:
            horizontal = self.get_horizontal_ground_vector()
            self.autopilot.target_direction = horizontal

        self.vessel.control.throttle = THROTTLE_OFF
        self.autopilot.disengage()

    def get_horizontal_ground_vector(self):
        retrograde = self.flight.retrograde
        horizontal = list(retrograde)
        horizontal[0] = 0.0
        return horizontal

    def prep_landing(self):
        print("Setting retrograde point")
        self.enable_retrograde_point()
        print("Retracting solar panels")
        self.retract_solar_panels()
        print("Activating parachutes")
        self.activate_parachutes()
        print("Deploying landing gear")
        self.deploy_landing_gear()

    def enable_retrograde_point(self):
        self.autopilot.disengage()
        self.autopilot.sas = True
        time.sleep(1.0)
        self.autopilot.sas_mode = self.conn.space_center.SASMode.retrograde
        #self.autopilot.engage()
        #self.autopilot.wait()

    def retract_solar_panels(self):
        for panel in self.vessel.parts.solar_panels:
            panel.deployed = False
        
    def activate_parachutes(self):
        self.vessel.control.activate_next_stage()
        # TODO this doesn't work for some reason?
        #self.vessel.control.parachutes = True

    def deploy_landing_gear(self):
        self.vessel.control.gear = True

    def do_landing_burn(self):
        orbit = self.vessel.orbit
        flight = self.vessel.flight()

        mass = self.vessel.mass
        max_thrust = self.vessel.max_thrust
        grav_accel = self.vessel.orbit.body.surface_gravity

        while True:
            d = flight.surface_altitude
            v = orbit.speed
            a = (max_thrust - grav_accel * mass) / mass  # Ship decelleration is due to thrust and the planets gravity
            breaking = 0.5 * v**2.0 / a

            # When we are at the intersection of velocity and distance to do a suicide burn, execute the burn
            if((d - breaking) < BREAKING_THRESHOLD) and d < MIN_BURN_ALTITUDE:
                break

        self.vessel.control.throttle = MAX_THROTTLE

        while abs(self.vessel.flight(self.vessel.orbit.body.reference_frame).vertical_speed) > LANDING_SPEED:
             pass

        grav_force = grav_accel * mass
        fraction = grav_force / max_thrust
        self.vessel.control.throttle  = fraction

        while self.vessel.situation != self.conn.space_center.VesselSituation.landed:
            pass
        self.vessel.control.throttle = THROTTLE_OFF

    def do_enter_aerobreak(self, aerobrake_to):
        self.warp_to_apoapsis()
        self.autopilot.sas = True
        self.autopilot.sas_mode = self.conn.space_center.SASMode.retrograde
        self.autopilot.wait()
        self.vessel.control.throttle = MAX_THROTTLE
        while self.vessel.orbit.periapsis_altitude > aerobrake_to:
            pass
        self.vessel.control.throttle = THROTTLE_OFF

    def warp_to_apoapsis(self):
        current_time = self.conn.space_center.ut
        time_to_apoapsis = self.vessel.orbit.time_to_apoapsis + current_time
        self.conn.space_center.warp_to(time_to_apoapsis)

    def wait_for_equatorial_pass(self):
        approaching = False
        last_latitude = abs(self.vessel.position(self.vessel.orbit.body.orbital_reference_frame)[2])
        while True:
            position = self.vessel.position(self.vessel.orbit.body.orbital_reference_frame)[2]
            print("Position: ", position, approaching, end='\r')
            if not approaching and abs(position) < last_latitude:
                approaching = True
            if approaching and abs(position) > last_latitude:
                self.conn.space_center.rails_warp_factor = 0
                return
            last_latitude = abs(position)

    def plot_orbit(self):
        Cd = DRAG_COEFF
        mu = self.vessel.orbit.body.gravitational_parameter
        R = self.vessel.orbit.body.equatorial_radius
        Bc = self.flight.ballistic_coefficient
        mass = self.vessel.mass
        H = self.vessel.orbit.body.atmosphere_depth
        T_0 = 250.15 # average for Duna
        print("Aerovals: ", Cd, mu, R, Bc, mass, H, T_0)
        aerobrake_calculator = aerobraking.AerobrakeCalculator(Cd, mu, R, VESSEL_SURFACE_AREA,
                                                               mass, H, T_0)
        position = self.vessel.position(self.vessel.orbit.body.orbital_reference_frame)
        velocity = self.vessel.velocity(self.vessel.orbit.body.orbital_reference_frame)
        print(position, velocity)
        aerobrake_calculator.calculate_aero(position, velocity)
    
    def calculator_generator(self):
        Cd = DRAG_COEFF
        mu = self.vessel.orbit.body.gravitational_parameter
        R = self.vessel.orbit.body.equatorial_radius
        Bc = self.flight.ballistic_coefficient
        mass = self.vessel.mass
        H = self.vessel.orbit.body.atmosphere_depth
        T_0 = 250.15 # average for Duna
        aerobrake_calculator = aerobraking.AerobrakeCalculator(Cd, mu, R, VESSEL_SURFACE_AREA,
                                                               mass, H, T_0)
        return aerobrake_calculator

    def calc_periapsis_change(self, change):
        apoapsis = self.orbit.apoapsis
        periapsis = self.orbit.periapsis
        new_periapsis = self.orbit.periapsis + change
        mu = self.orbit.body.gravitational_parameter
        deltav = orbital_calculations.apsis_change_dv(mu, apoapsis, periapsis, new_periapsis)
        return deltav

    def find_optimal_burn(self):
        self.warp_to_apoapsis()
        position = self.vessel.position(self.vessel.orbit.body.orbital_reference_frame)
        velocity = self.vessel.velocity(self.vessel.orbit.body.orbital_reference_frame)
        apoapsis = self.orbit.apoapsis
        periapsis = self.orbit.periapsis
        aerobreak_calculator = self.calculator_generator()
        passes, goal_periapsis = aerobreak_calculator.optimal_aero(position, velocity, apoapsis, periapsis, 500)

if __name__ == "__main__":
    sc = AutoPilot()
    sc.wait_for_equatorial_pass()
    sc.horizontal_burn()
    sc.prep_landing()
    sc.do_landing_burn()
