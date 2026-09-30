"""Winding layout of an m-phase AC stator winding for AEDT FEM models.

Assigns a phase letter and polarity to each slot's coil side, for single-
or double-layer windings with integer or fractional slots per pole per
phase q = n/c (c must be coprime with the number of phases). Odd and even
phase counts are supported; an even-phase winding is treated as several
symmetrical sub-phase sets shifted against each other (e.g. six phases =
two 3-phase sets 30 el. deg apart).

Besides the slot assignment, the layout provides:
  - the electrical angle of each phase,
  - the winding factor (pitch x distribution), with a warning below 0.9,
  - p2_min_fem, the minimum pole count of the FEM sector (odd values mean
    an antiperiodic boundary),
  - the electrical axis of phase A in the model,
  - the coil pitch, auto-selected when not given.

assign_slot_currents_to_aedt then creates AEDT coils and current windings
from a layout.
"""

import warnings
from dataclasses import dataclass
from math import ceil, gcd, pi, sin
from fractions import Fraction
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WindingLayout:
    coil_pitch: int
    phase: List[str]
    polarity: List[int]
    phase_order: List[str]
    winding_factor: float
    phase_angles: Dict[str, float]  # electrical angle [deg] of each phase, e.g. {"A": 0.0, "B": -30.0, ...}
    p2_min_fem: int  # minimum pole count of the FEM sector (odd allowed - antiperiodic boundary)
    phase_a_axis: float  # electrical axis [deg, 0..180) of phase A in the model
    phase_back: Optional[List[str]] = None
    polarity_back: Optional[List[int]] = None

def _phase_axis_tables(phases: int):
    """
    Returns (phase_letters, phase_signs, phase_angles): the phase letter and
    +-1 marker for each of the `phases` phase-group positions used by the
    main assignment loop, fixed by sorting all 2*phases physical+mathematical
    phase axes by electrical angle (descending) and keeping the first
    `phases`; phase_angles maps each phase letter to its electrical angle.

    An even phase count m is treated as x symmetrical (m/x)-phase sets, x
    being the largest power of two dividing m, with neighbouring sets
    shifted by 360/(2*m) electrical degrees. E.g. six phases are two
    symmetrical 3-phase sets (ACE and BDF) shifted by 30 degrees.
    """
    if phases % 2 == 0:
        phase_sets = 1
        reduced_phases = phases
        gcd_value = 2
        while gcd_value == 2:
            gcd_value = gcd(reduced_phases, 2)
            phase_sets *= gcd_value
            reduced_phases //= gcd_value

        phase_shift_step = -360.0 / reduced_phases
        phase_angles_base = [i * phase_shift_step for i in range(reduced_phases)]
        phase_angle_matrix = []
        for set_index in range(phase_sets):
            set_shift = -360.0 / (2 * phases) * set_index
            phase_angle_matrix.append([v + set_shift for v in phase_angles_base])
        # Flatten column-major (set index varies fastest).
        phase_angles = [
            phase_angle_matrix[row][col]
            for col in range(reduced_phases)
            for row in range(phase_sets)
        ]
        phase_angles.sort(reverse=True)
    else:
        phase_shift_step = -360.0 / phases
        phase_angles = [i * phase_shift_step for i in range(phases)]

    # phase_angles[i] is the axis of phase chr(65 + i) (A, B, C, ...)
    phase_angle_by_letter = {chr(65 + i): angle + 0.0 for i, angle in enumerate(phase_angles)}

    phase_angles_back = []
    for angle in phase_angles:
        angle_back = angle + 180.0
        if angle_back > 0:
            angle_back -= 360.0
        phase_angles_back.append(angle_back)

    phase_angles_all = phase_angles + phase_angles_back
    phase_signs = [1] * phases + [-1] * phases
    phase_letter_codes = [chr(65 + i) for i in range(phases)] * 2

    order = sorted(range(2 * phases), key=lambda i: phase_angles_all[i], reverse=True)
    phase_letters_sorted = [phase_letter_codes[i] for i in order]
    phase_signs_sorted = [phase_signs[i] for i in order]

    return phase_letters_sorted[:phases], phase_signs_sorted[:phases], phase_angle_by_letter


def compute_winding_layout(
    phases: int,
    slots: int,
    pole_pairs: int,
    coil_pitch: Optional[int] = None,
    layers: int = 1,
) -> WindingLayout:
    """Compute per-slot phase letter and polarity for an m-phase AC winding.

    `coil_pitch` is given in slots; None or 0 auto-selects the pitch closest
    to (2m-1)/(2m) of the pole pitch (odd for a single layer). For
    layers == 1, each slot holds one coil side; `phase_back`/`polarity_back`
    are populated only for layers == 2 and describe the second (inner/bottom)
    radial coil side per slot.
    """
    if layers not in (1, 2):
        raise ValueError("layers must be 1 or 2")

    pole_count = 2 * pole_pairs
    if slots % (phases) != 0:
        raise ValueError(
            f"slots ({slots}) must be divisible by number of phases ({phases})"
        )
    slots_ppp = Fraction(slots, pole_count * phases)  # q, as an exact fraction n/c
    slots_ppp_numer = slots_ppp.numerator  # n
    slots_ppp_denom = slots_ppp.denominator  # c
    if gcd(slots_ppp_denom, phases) != 1:
        raise ValueError(
            f"c ({slots_ppp_denom}) and the number of phases ({phases}) must be "
            "coprime for a symmetric winding"
        )
    pole_pitch_slots = slots / pole_count # t_pd

    # effective_n (coil sides per phase group, used for the distribution
    # factor) is derived from n, not q; it and c together fix p2_min - the minimum pole count for which the winding
    # pattern closes periodically. The model's actual pole count must be an
    # integer multiple of p2_min.
    if layers == 2:
        effective_n = slots_ppp_numer
        p2_min = 2 * slots_ppp_denom if slots_ppp_denom % 2 == 1 else slots_ppp_denom
        p2_min_fem = slots_ppp_denom
    elif slots_ppp_numer % 2 == 0:
        effective_n = slots_ppp_numer // 2
        p2_min = 2 * slots_ppp_denom if slots_ppp_denom % 2 == 1 else slots_ppp_denom
        p2_min_fem = slots_ppp_denom
    else:
        effective_n = slots_ppp_numer
        p2_min = 2 * slots_ppp_denom
        p2_min_fem = 2*slots_ppp_denom
    if pole_count % p2_min != 0:
        raise ValueError(
            f"pole count ({pole_count}) must be a multiple of the minimum pole "
            f"count {p2_min} required for this winding (phases={phases}, "
            f"slots={slots}, layers={layers}) to close periodically"
        )

    if not coil_pitch:
        optimal_pitch_ratio = (2 * phases - 1) / (2 * phases)
        optimal_coil_pitch = optimal_pitch_ratio * pole_pitch_slots
        coil_pitch = round(optimal_coil_pitch)
        if layers == 1 and coil_pitch % 2 == 0:
            coil_pitch = coil_pitch - 1 if coil_pitch > optimal_coil_pitch else coil_pitch + 1
    if layers == 1 and coil_pitch % 2 == 0:
        raise ValueError(f"single-layer winding requires an odd coil pitch, got {coil_pitch}")

    phase_letters, phase_signs, phase_angles = _phase_axis_tables(phases)

    single_layer_flag = layers % 2  # 1 = single layer (needs gap-fill), 0 = double layer
    slot_step = 1 + single_layer_flag

    phase: List[Optional[str]] = [None] * slots
    polarity: List[Optional[int]] = [None] * slots
    front_slot_indices: List[int] = []

    # cell_index walks the technological-scheme grid (step slot_step*c);
    # slot_index walks the physical slots (step slot_step) - the two only
    # stay in lockstep when c == 1.
    cell_step = slot_step * slots_ppp_denom
    pole_row = 1
    sign = 1
    cell_index = 1
    slot_index = 0
    while cell_index <= pole_count * phases * slots_ppp_numer:
        while cell_index > phases * slots_ppp_numer * pole_row:
            pole_row += 1
            sign = -sign
        phase_group = ceil(
            (cell_index - (pole_row - 1) * phases * slots_ppp_numer) / slots_ppp_numer
        )  # 1..phases
        phase[slot_index] = phase_letters[phase_group - 1]
        polarity[slot_index] = sign * phase_signs[phase_group - 1]
        front_slot_indices.append(slot_index)
        slot_index += slot_step
        cell_index += cell_step

    phase_back: Optional[List[str]] = None
    polarity_back: Optional[List[int]] = None

    if layers == 1:
        # For a single layer the loop above fills every other slot only;
        # the remaining slots are the back coil sides, related to the front
        # ones by the coil pitch: back[s] = -front[(s - coil_pitch) mod slots].
        for slot in range(slots):
            if phase[slot] is None:
                source = (slot - coil_pitch) % slots
                phase[slot] = phase[source]
                polarity[slot] = -polarity[source]
    else:
        phase_back = [None] * slots
        polarity_back = [None] * slots
        for slot in range(slots):
            source = (slot - coil_pitch) % slots
            phase_back[slot] = phase[source]
            polarity_back[slot] = -polarity[source]

    pitch_ratio = coil_pitch / pole_pitch_slots
    pitch_factor = sin(pitch_ratio * pi / 2)
    distribution_factor = sin(pi / (2 * phases)) / (
        effective_n * sin(pi / (2 * phases * effective_n))
    )
    winding_factor = pitch_factor * distribution_factor
    if winding_factor < 0.9:
        warnings.warn(f"winding factor is low: {winding_factor:.3g} (< 0.9)")

    phase_order = [chr(65 + i) for i in range(phases)]

    # Phase A axis: mean (mod 180) electrical angle of phase A's front coil
    # sides, plus half the coil pitch (front side -> coil axis) and half a
    # slot pitch (index 0 sits on a tooth, not a slot center - the base coil
    # rectangle is rotated by half a slot pitch before being duplicated
    # around the machine).
    slot_pitch_el = 360.0 * pole_pairs / slots
    a_angles = [(i * slot_pitch_el) % 180.0 for i in front_slot_indices if phase[i] == "A"]
    if not a_angles:
        raise ValueError("phase 'A' has no front coil side in this winding")
    front_a_axis = sum(a_angles) / len(a_angles)
    phase_a_axis = (front_a_axis + coil_pitch * slot_pitch_el / 2.0 + slot_pitch_el / 2.0) % 180.0

    return WindingLayout(
        coil_pitch=coil_pitch,
        phase=phase,
        polarity=polarity,
        phase_order=phase_order,
        winding_factor=winding_factor,
        phase_angles=phase_angles,
        p2_min_fem=p2_min_fem,
        phase_a_axis=phase_a_axis,
        phase_back=phase_back,
        polarity_back=polarity_back,
    )


def assign_slot_currents_to_aedt(
    m2d,
    coil_objects: List[str],
    layout: WindingLayout,
    phase_currents: Dict[str, str],
    turns_var: str = "Nc",
    parallel_paths_var: str = "ParallelPaths",
) -> Dict[str, str]:
    """Assign AEDT coils and windings from a WindingLayout.

    Replaces the hand-written assign_coil/assign_winding/add_winding_coils
    blocks. `coil_objects` must be in the same slot order as
    `layout.phase`/`layout.polarity` (i.e. modeler.get_objects_w_string(...)
    right after duplicate_around_axis). Returns {phase_letter: winding_name}
    for the windings that were created.
    """
    coil_names_by_phase: Dict[str, List[str]] = {p: [] for p in layout.phase_order}

    for i, obj in enumerate(coil_objects):
        letter = layout.phase[i]
        polarity = "Positive" if layout.polarity[i] > 0 else "Negative"
        cs_name = f"CS{i + 1}"
        m2d.assign_coil(assignment=[obj], conductors_number=turns_var, polarity=polarity, name=cs_name)
        coil_names_by_phase[letter].append(cs_name)

    winding_names: Dict[str, str] = {}
    for letter, cs_list in coil_names_by_phase.items():
        if not cs_list or letter not in phase_currents:
            continue
        winding_name = f"Phase{letter}"
        m2d.assign_winding(
            assignment=None,
            winding_type="Current",
            is_solid=False,
            current=phase_currents[letter],
            parallel_branches=parallel_paths_var,
            name=winding_name,
        )
        m2d.add_winding_coils(assignment=winding_name, coils=cs_list)
        winding_names[letter] = winding_name

    return winding_names


if __name__ == "__main__":
    layout_6ph = compute_winding_layout(phases=6, slots=48, pole_pairs=5, coil_pitch=5, layers=1)
    layout_6ph_v2 = compute_winding_layout(phases=6, slots=48, pole_pairs=2, coil_pitch=11, layers=1)
    
    layout_3ph = compute_winding_layout(phases=3, slots=6, pole_pairs=2, coil_pitch=1, layers=2)
    layout_5ph = compute_winding_layout(phases=5, slots=40, pole_pairs=2, coil_pitch=9, layers=1)

    print("All self-tests passed.")
