import numpy as np


def get_semimajor_axis(apoapsis, periapsis):
    return (apoapsis + periapsis) / 2.0

def visviva(mu, position, apoapsis, periapsis):
    a = get_semimajor_axis(apoapsis, periapsis)
    v = np.sqrt(mu * ( 2 / position - 1 / a))
    return v

def apsis_change_dv(mu, a1, p1, p2):
    v1 = visviva(mu, a1, a1, p1)
    v2 = visviva(mu, a1, a1, p2)

    deltav = v2 - v1
    return deltav