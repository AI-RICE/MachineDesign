from machine_design.winding_layout_JL import compute_winding_layout

expect_5 = {  # true value of 5f_40s from winding.py
    "A": {(0, 1), (9, -1)},
    "B": {(3, 1), (4, 1)},
    "C": {(7, 1), (8, 1)},
    "D": {(1, -1), (2, -1)},
    "E": {(5, -1), (6, -1)},
}

expect_3 = {  # true value of 3f_36s from winding.py
    "A": {(0, 1), (1, 1), (2, 1)},
    "B": {(6, 1), (7, 1), (8, 1)},
    "C": {(3, -1), (4, -1), (5, -1)},
}


def sector_map(layout, n):
    out = {}
    for slot in range(n):
        out.setdefault(layout.phase[slot], set()).add((slot, layout.polarity[slot]))
    return out


layout_5 = compute_winding_layout(phases=5, slots=40, pole_pairs=2, coil_pitch=9, layers=1)
got_5 = sector_map(layout_5, 10)
assert got_5 == expect_5, f"5f mismatch:{got_5}"
print("5f p2_min_fem:", layout_5.p2_min_fem)

layout_3 = compute_winding_layout(phases=3, slots=36, pole_pairs=2, coil_pitch=9, layers=1)
got_3 = sector_map(layout_3, 9)
assert got_3 == expect_3, f"3f mismatch:{got_3}"
print("3f p2_min_fem:", layout_3.p2_min_fem)
