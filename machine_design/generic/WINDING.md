# Winding layout

`machine_design/generic/winding.py` computes the phase and polarity
layout of each slot, and assigns it to an AEDT model.
It covers `compute_winding_layout` and `assign_slot_currents_to_aedt`.

## Basic usage

```python
compute_winding_layout(phases, slots, pole_pairs, coil_pitch=None, layers=1)
```

- phases, slots, pole_pairs: the machine's basic parameters.
- coil_pitch: coil span in slots.
- layers: 1 for a single layer winding (default), 2 for double layer.

Example, using the synrm_3f_36s motor (3 phases, 36 slots, 2 pole pairs, coil_pitch=9):

```python
from machine_design.generic.winding import compute_winding_layout
layout = compute_winding_layout(phases=3, slots=36, pole_pairs=2, coil_pitch=9, layers=1)
```

The first 9 slots of layout.phase and layout.polarity give:
```
phase[:9]    = ['A','A','A','C','C','C','B','B','B']
polarity[:9] = [1,  1,  1,  -1,  -1,  -1,  1,  1,  1]
```

Summary:
Slots 0-2 carry phase A (positive), slots 3-5 carry phase C (negative),
and slots 6-8 carry phase B (positive).

## Meaning of the outputs
compute_winding_layout returns a WindingLayout. For the example above:

| Field | Value | Meaning |
|---|---|---|
| `coil_pitch` | `9` | Coil span in slots, including the automatically selected value if applicable. |
| `phase` | `['A','A','A','C',...]` | Phase letter of each slot's front coil side. |
| `polarity` | `[1,1,1,-1,...]` | Current direction of each slot's front coil side. |
| `phase_order` | `['A','B','C']` | The phase letters used in this winding. |
| `winding_factor` | `0.9598` | Pitch times distribution factor (0-1) (a warning is raised below 0.9). |
| `phase_angles` | `{'A':0.0,'B':-120.0,'C':-240.0}` | Electrical angle of each phase. |
| `p2_min_fem` | `1` | Minimum pole count for an FEM sector to close periodically (odd means the sector needs an antiperiodic boundary). |
| `phase_a_axis` | `120.0` | Electrical axis of phase A in the model (0-180 deg) (used to compute `InitPos`). |
| `phase_back` / `polarity_back` | `None` | Second (back) coil side per slot (only set for `layers=2`, `None` for a single layer). |


## Double layer windings (layers=2)
With layers=2, each slot holds two radially stacked coil slides: a front
side, described by phase/polarity as before, and a back side, described
by phase_back/polarity_back (same format, same slot order). The back side
is the front side shifted by coil_pitch slots, with reversed polarity.

```python
layout = compute_winding_layout(phases=3, slots=36, pole_pairs=2, coil_pitch=9, layers=2)
```

The first 9 slots of layout.phase/layout.polarity (front) and
layout.phase_back/layout.polarity_back (back):
```
phase[:9] = ['A','A','A','C','C','C','B','B','B']
phase_back[:9] = ['A','A','A','C','C','C','B','B','B']
polarity[:9] = [1,  1,  1,  -1,  -1,  -1,  1,  1,  1]
polarity_back[:9] = [1,  1,  1,  -1,  -1,  -1,  1,  1,  1]
```

Front and back happen to be identical here only because `coil_pitch=9`
is a full pole pitch. A shorter pitch would make them differ.

To assign a double layer layout to AEDT, build both coil sides as separate
named objects (one set per layer) and pass the second layer's object names
as `coil_objects_back` to `assign_slot_currents_to_aedt`:

assign_slot_currents_to_aedt(m2d, coil_objects, layout, phase_currents, coil_objects_back=coil_objects_back)

## Relation to coil_pitch/belt_offset

The project's old winding code used belt_offset instead of coil_pitch.
Let pole_pitch_slots = slots/(2*pole_pairs) (the pole pitch in slots):
- If pole_pitch_slots is odd: coil_pitch = pole_pitch_slots, belt_offset = 0.
- If pole_pitch_slots is even: coil_pitch = pole_pitch_slots-1 with belt_offset=1, or 
coil_pitch = pole_pitch_slots+1 with belt_offset=0.
