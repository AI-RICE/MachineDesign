"""RadialSpline round-trip fidelity decomposition (no FEA).

Pins WHY re-encoding known-good Hackl designs degrades them ~9-12% HV (E7/E8):
is it (B) the B-spline resolution K, or (A) the repair's structural minimums
(t_rib/t_shaft/min_air) being more conservative than the real Hackl designs?

A. STRUCTURAL GAPS: measure the actual minimum iron gaps in good Hackl designs
   (inter-barrier, shaft-side) and compare to the repair minimums we imposed.
B. ROUND-TRIP DECOMPOSITION: for the same designs, fit_barriers at several K and
   split the boundary error into
     fit error      = raw B-spline(K) vs Hackl   (resolution lever),
     repair distort  = repaired vs raw B-spline   (repair-constraint lever),
   plus the total round-trip error.

  .venv/bin/python notebooks/radialspline_fidelity.py --results-root ../MachineDesign/results
"""

import argparse
import os
import sys

import numpy as np
import torch
from shapely.geometry import LineString

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.utils.multi_objective.pareto import is_non_dominated  # noqa: E402

from machine_design.fea_emulator import load_fea_designs  # noqa: E402
from machine_design.generators import (  # noqa: E402
    HacklGenerator_3BrokenLines,
    HacklGenerator_OneLambda,
    HacklGenerator_SixLambdas,
    RadialSplineGenerator,
)
from machine_design.lumped.geometry import REFERENCE_MACHINE  # noqa: E402

GENS = [("OneLambda", HacklGenerator_OneLambda), ("SixLambdas", HacklGenerator_SixLambdas),
        ("ThreeBrokenLines", HacklGenerator_3BrokenLines)]


def envelope(poly, tg):
    """Closed barrier polygon -> (outer r, inner r) on theta grid tg (deg)."""
    r = np.linalg.norm(poly, axis=1)
    th = np.degrees(np.arctan2(poly[:, 1], poly[:, 0]))
    o = np.full(len(tg), np.nan); i = np.full(len(tg), np.nan)
    dt = tg[1] - tg[0]
    idx = np.clip(((th - tg[0]) / dt + 0.5).astype(int), 0, len(tg) - 1)
    for k in range(len(tg)):
        rr = r[idx == k]
        if rr.size:
            o[k] = rr.max(); i[k] = rr.min()
    return o, i


def good_hackl_designs(results_root, per_gen=12):
    """Pareto-front (high-T/low-ripple) Hackl designs per parameterisation."""
    out = []
    for short, cls in GENS:
        d = load_fea_designs(short, results_root=results_root, constrained=None)
        f = np.column_stack([d.T_mean, -d.T_ripple / 100.0])
        nd = np.where(is_non_dominated(torch.tensor(f)).numpy())[0]
        # also top-T to ensure the high-torque region is covered
        topT = np.argsort(d.T_mean)[-per_gen:]
        idx = np.unique(np.concatenate([nd, topT]))[:per_gen * 2]
        hk = cls(REFERENCE_MACHINE, r_stator_end=0.7, offset=0.35)
        for j in idx:
            hk.set_parameters(hk.X_to_params(np.asarray(d.X[j], float)))
            out.append((short, hk.generate_barriers(), float(d.T_mean[j]), float(d.T_ripple[j])))
    return out


def part_A_gaps(designs, gen):
    """Actual min iron gaps in Hackl designs vs imposed repair minimums."""
    inter, shaft, surf = [], [], []
    for _, bars, _, _ in designs:
        polys = sorted(bars, key=lambda b: np.linalg.norm(b, axis=1).min())
        ls = [LineString(b) for b in polys]
        for a in range(len(ls) - 1):
            inter.append(ls[a].distance(ls[a + 1]))
        shaft.append(np.linalg.norm(polys[0], axis=1).min() - gen.r_min)
        surf.append(gen.r_max - max(np.linalg.norm(b, axis=1).max() for b in bars))
    inter, shaft, surf = np.array(inter), np.array(shaft), np.array(surf)
    print("\n=== A. Actual Hackl structural gaps vs imposed repair minimums ===")
    print(f"  inter-barrier iron gap : min={inter.min():.3f}  median={np.median(inter):.3f}  "
          f"mean={inter.mean():.3f} mm   | repair t_rib   = {gen.t_rib}")
    print(f"  shaft-side iron gap    : min={shaft.min():.3f}  median={np.median(shaft):.3f}  "
          f"mean={shaft.mean():.3f} mm   | repair t_shaft = {gen.t_shaft}")
    print(f"  surface-side iron gap  : min={surf.min():.3f}  median={np.median(surf):.3f}  "
          f"mean={surf.mean():.3f} mm   | repair t_bridge= {gen.t_bridge}")
    print(f"  -> inter-barrier gaps below t_rib={gen.t_rib}: "
          f"{100*(inter < gen.t_rib).mean():.0f}% of adjacencies "
          f"(these get pushed apart by repair)")


def raw_envelope(gen, theta_lo, theta_hi, c_out, c_in, tg):
    """Raw B-spline boundaries (pre-repair) on theta grid tg, within the span."""
    m = (tg >= theta_lo) & (tg <= theta_hi)
    s = np.clip((tg[m] - theta_lo) / max(theta_hi - theta_lo, 1e-9), 0, 1)
    B = gen._basis_at(s, gen.knots, gen.spline_k)
    o = np.full(len(tg), np.nan); i = np.full(len(tg), np.nan)
    o[m] = B @ np.asarray(c_out, float); i[m] = B @ np.asarray(c_in, float)
    return o, i


def part_B_decomp(designs, Ks):
    """fit error (resolution) vs repair distortion, by K."""
    print("\n=== B. Round-trip error decomposition (RMS over boundaries, mm) ===")
    print(f"{'K':>4} | {'fit err':>8} | {'fit@ends':>9} | {'fit@mid':>8} | {'repair':>7} | {'total':>7}")
    tg = np.linspace(6, 84, 160)
    ends = (tg < 25) | (tg > 65); mid = (tg >= 35) & (tg <= 55)
    for K in Ks:
        gen = RadialSplineGenerator(REFERENCE_MACHINE, K=K)
        fit_e, rep_e, tot_e, fend, fmid = [], [], [], [], []
        for _, bars, _, _ in designs:
            X = gen.fit_barriers(bars)
            params = gen.X_to_params(X)
            gen.set_parameters(X)
            rep_bars = gen.generate_barriers()
            hk_sorted = sorted(bars, key=lambda b: np.linalg.norm(b, axis=1).min())
            rep_sorted = sorted(rep_bars, key=lambda b: np.linalg.norm(b, axis=1).min())
            for (tlo, thi, co, ci), hb, rb in zip(params, hk_sorted, rep_sorted):
                ho, hi_ = envelope(hb, tg)
                ro_raw, ri_raw = raw_envelope(gen, tlo, thi, co, ci, tg)
                ro_rep, ri_rep = envelope(rb, tg)
                mask = ~(np.isnan(ho) | np.isnan(ro_raw) | np.isnan(ro_rep) |
                         np.isnan(hi_) | np.isnan(ri_raw) | np.isnan(ri_rep))
                if mask.sum() < 5:
                    continue
                def rms(a, b, m=mask):
                    mm = m & mask
                    return np.sqrt(np.mean((a[mm] - b[mm]) ** 2)) if mm.sum() else np.nan
                fit_e.append(0.5 * (rms(ro_raw, ho) + rms(ri_raw, hi_)))
                rep_e.append(0.5 * (rms(ro_rep, ro_raw) + rms(ri_rep, ri_raw)))
                tot_e.append(0.5 * (rms(ro_rep, ho) + rms(ri_rep, hi_)))
                fend.append(0.5 * (rms(ro_raw, ho, ends) + rms(ri_raw, hi_, ends)))
                fmid.append(0.5 * (rms(ro_raw, ho, mid) + rms(ri_raw, hi_, mid)))
        print(f"{K:>4} | {np.mean(fit_e):>8.3f} | {np.nanmean(fend):>9.3f} | {np.nanmean(fmid):>8.3f} | "
              f"{np.mean(rep_e):>7.3f} | {np.mean(tot_e):>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="../MachineDesign/results")
    args = ap.parse_args()
    designs = good_hackl_designs(args.results_root)
    print(f"good Hackl designs sampled: {len(designs)} "
          f"({', '.join(f'{s}:{sum(1 for d in designs if d[0]==s)}' for s,_ in GENS)})")
    gen18 = RadialSplineGenerator(REFERENCE_MACHINE, K=18)
    part_A_gaps(designs, gen18)
    part_B_decomp(designs, Ks=[18, 33, 48])


if __name__ == "__main__":
    main()
