"""Grid sweep comparison of winding.py (old) and winding_layout_JL.py (new) across
(phases, slots, pole_pairs) combinations.
Run with:
    python scripts/match_winding_params_grid.py

"""

import warnings

warnings.filterwarnings("ignore")

from machine_design.generic.compute_initpos import compute_initpos
from machine_design.generic.winding import sector_winding
from machine_design.winding_layout_JL import compute_winding_layout

phase_name = "ABCDEFGHIJK"


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
        for coil_pitch in range(1, int(pole_pitch_slots) + 2, 2):
            new = new_sector(phases, slots, pole_pairs, coil_pitch, n)
            if old == new:
                matches.append((belt_offset, coil_pitch))
    return matches


for phases in (3, 5, 7, 9, 11):
    for pole_pairs in (1, 2, 3, 4):
        for q in (1, 2, 3):
            # compare phases and polarity
            slots = q * 2 * pole_pairs * phases
            matches = find_matches(phases, slots, pole_pairs)
            assert matches, f"No match: phases={phases} slots={slots} pole_pairs={pole_pairs}"

            # verify InitPos between old and new code
            for belt_offset, coil_pitch in matches:
                initpos = compute_initpos(slots, pole_pairs, phases, belt_offset)
                layout = compute_winding_layout(phases, slots, pole_pairs, coil_pitch=coil_pitch, layers=1)
                initpos_from_new = (layout.phase_a_axis - 180) / pole_pairs
                assert abs(initpos_from_new - initpos) < 1e-6, f"InitPos mismatch: phases={phases} slots={slots} pole_pairs={pole_pairs}"

                # verify phase_angles
                for k, letter in enumerate(layout.phase_order):
                    expected = -360.0 * k / phases
                    assert abs(layout.phase_angles[letter] - expected) < 1e-6, f"phase_angles mismatch: phases={phases} letter={letter}"

print("all combinations matched")

# fractional slots
fractional_slots=[(3,12,5)]  #  (phases, slots, pole_pairs)
for phases,slots,pole_pairs in fractional_slots:
    try:
        sector_winding(slots,pole_pairs,phases,0)
        raise AssertionError(f"old supports fractional slots: phases={phases} slots={slots} pole_pairs={pole_pairs}")
    except ValueError:
        pass
    layout=compute_winding_layout(phases,slots,pole_pairs,layers=1)
    assert layout.winding_factor>0.9, f"low winding factor: phases={phases} slots={slots} pole_pairs={pole_pairs}"
