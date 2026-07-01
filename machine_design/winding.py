"""Parametric stator winding-layout generator (pure algorithm, no AEDT).

Produces the coil -> (phase, polarity) map for the one-pole, anti-periodic FEA
sector our machine models, for an arbitrary balanced integer-slot distributed
winding given (Q slots, p pole-pairs, m phases). It replaces the hardcoded
per-coil assignment in Design.assign_stator_coils / Design2.assign_stator_coils.

Convention (reverse-engineered from, and validated to exactly reproduce, the
project's 40-slot/5-phase and 36-slot/3-phase windings):
  - the FEA sector spans ONE pole, n = Q/(2p) slots = coils, with anti-periodic BC;
  - q = Q/(2*p*m) slots per phase belt (must be a positive integer);
  - coils are created in slot order: coil 0 = "Coil", coil k = "Coil_{k}";
  - belt index of coil k:  b = (k + 1) // q          (raw, half-slot-offset origin)
  - phase index:           ((m+1)//2 * b) mod m       (standard (m+1)/2 progression)
  - polarity:              +1 if b even else -1        (alternating belts; the wrap
                                                        coil flips via anti-periodicity)
"""


def coil_name(k: int) -> str:
    """AEDT object name for the k-th duplicated coil (0-based)."""
    return "Coil" if k == 0 else f"Coil_{k}"


def sector_winding(Q: int, p: int, m: int, belt_offset: int = 1):
    """Return a list (length n = Q/(2p)) of (phase_index 0..m-1, sign +/-1),
    one entry per sector coil in slot/creation order.

    belt_offset shifts the belt origin (a pure spatial rotation of the winding,
    immaterial to performance, absorbed by the rotor-alignment / current angle):
      - belt_offset=1 -> PhaseA centred on the d-axis with an anti-periodic wrap;
        reproduces the project's 5-phase (40-slot) winding. Default / preferred.
      - belt_offset=0 -> belts start at slot 0; reproduces the legacy 3-phase
        (36-slot) base winding.
    New machines should use the default (1) and align the dq reference to it."""
    if Q % (2 * p * m) != 0:
        raise ValueError(
            f"non-integer slots/pole/phase: Q={Q}, 2*p*m={2 * p * m} "
            f"(q=Q/(2pm) must be a positive integer for this winding)")
    n = Q // (2 * p)          # coils in the one-pole sector
    q = Q // (2 * p * m)      # slots per belt
    step = (m + 1) // 2
    out = []
    for k in range(n):
        b = (k + belt_offset) // q
        out.append(((step * b) % m, 1 if b % 2 == 0 else -1))
    return out


def phase_groups(Q: int, p: int, m: int, belt_offset: int = 1):
    """Return m lists; phase j -> [(coil_name, polarity_str), ...] for driving
    assign_coil / add_winding_coils generically."""
    groups = [[] for _ in range(m)]
    for k, (ph, sg) in enumerate(sector_winding(Q, p, m, belt_offset)):
        groups[ph].append((coil_name(k), "Positive" if sg > 0 else "Negative"))
    return groups


if __name__ == "__main__":
    PHASE = "ABCDEFG"
    # --- validation: must reproduce the existing hardcoded windings exactly ---
    def asmap(Q, p, m, off):
        g = {}
        for k, (ph, sg) in enumerate(sector_winding(Q, p, m, off)):
            g.setdefault(PHASE[ph], set()).add((k, sg))
        return g

    EXPECT_5 = {  # design2.assign_stator_coils  (CS_i -> Coil_{i-1} -> slot i-1)
        "A": {(0, 1), (9, -1)}, "B": {(3, 1), (4, 1)}, "C": {(7, 1), (8, 1)},
        "D": {(1, -1), (2, -1)}, "E": {(5, -1), (6, -1)}}
    EXPECT_3 = {  # design.assign_stator_coils  (PhaseA=CS1-3, B=CS7-9, C=CS4-6)
        "A": {(0, 1), (1, 1), (2, 1)}, "B": {(6, 1), (7, 1), (8, 1)},
        "C": {(3, -1), (4, -1), (5, -1)}}
    assert asmap(40, 2, 5, 1) == EXPECT_5, "5-phase (offset 1) mismatch"
    assert asmap(36, 2, 3, 0) == EXPECT_3, "3-phase (offset 0) mismatch"
    print("OK  reproduces BOTH hardcoded windings exactly: 5-phase 40-slot (offset 1), 3-phase 36-slot (offset 0)")

    def show(Q, p, m):
        lay = sector_winding(Q, p, m)
        bins = {}
        for k, (ph, sg) in enumerate(lay):
            bins.setdefault(PHASE[ph], []).append(f"{coil_name(k)}{'+' if sg > 0 else '-'}")
        q = Q // (2 * p * m)
        print(f"\n(Q={Q}, p={p}, m={m})  sector coils={len(lay)}  q={q}")
        for ph in sorted(bins):
            print(f"   Phase {ph}: {bins[ph]}")

    show(40, 2, 5)   # current 5-phase
    show(36, 2, 3)   # current 3-phase
    show(60, 2, 3)   # common-stator target, 3-phase
    show(60, 2, 5)   # common-stator target, 5-phase
