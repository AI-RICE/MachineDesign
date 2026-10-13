"""check winding.py and winding_layout_JL.py match
Run with:
python scripts/match_winding_params.py

"""

from machine_design.generic.winding import sector_winding
from machine_design.winding_layout_JL import compute_winding_layout

phase_name = "ABCDEFGH"


def old_sector(phases, slots, pole_pairs, belt_offset):
    return [(phase_name[ph], sg) for ph, sg in sector_winding(slots, pole_pairs, phases, belt_offset)]


def new_sector(phases, slots, pole_pairs, coil_pitch, n):
    layout = compute_winding_layout(phases, slots, pole_pairs, coil_pitch=coil_pitch, layers=1)
    return list(zip(layout.phase[:n], layout.polarity[:n]))


def find_matches(phases, slots, pole_pairs):
    n = slots // (2 * pole_pairs)
    pole_pitch_slots = slots / (2 * pole_pairs)
    matches = []
    for belt_offset in range(phases):
        old = old_sector(phases, slots, pole_pairs, belt_offset)
        for coil_pitch in range(1, int(pole_pitch_slots) + 4, 2):
            try:
                new = new_sector(phases, slots, pole_pairs, coil_pitch, n)
            except ValueError:
                continue
            if old == new:
                matches.append((belt_offset, coil_pitch))
    return matches


configs = [(5, 40, 2), (3, 36, 2), (3, 60, 2), (5, 60, 2), (3, 36, 3)]

for phases, slots, pole_pairs in configs:
    print(f"phases={phases} slots={slots} pole_pairs={pole_pairs}: {find_matches(phases, slots, pole_pairs)}")
