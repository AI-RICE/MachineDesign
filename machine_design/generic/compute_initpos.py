import math

from machine_design.generic.winding import sector_winding


def compute_initpos(Q: int, p: int, m: int, belt_offset: int):
    out1 = sector_winding(Q, p, m, belt_offset)
    xa = ya = 0.0
    for k, (phase_index, sign) in enumerate(out1):
        if phase_index != 0:  # only need phase A
            continue
        theta_e_deg = (0.5 + k) * (360.0 * p / Q)
        if sign > 0:
            eff_deg = theta_e_deg
        else:
            eff_deg = theta_e_deg + 180.0
        xa += math.cos(math.radians(eff_deg))
        ya += math.sin(math.radians(eff_deg))
    gamma_e_deg = math.degrees(math.atan2(ya, xa)) % 360
    return (gamma_e_deg - 90.0) / p
