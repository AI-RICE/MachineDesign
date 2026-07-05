"""One-solve probe: confirm design.compute() returns the flux linkages the voltage
constraint needs (Flux_d1/q1/d3/q3). If missing, gen-3's flux extraction would silently
drop every solve. Usage: python flux_probe.py"""
import numpy as np
import gen2  # noqa: F401 (parity with worker imports)
import h0h1_par as P
import h0h1_study as H

design = P.open_isolated_design("fluxprobe", 91, "2024.2", slots=60, phases=5)
gen = P.make_generator(design, False); lb, ub = H.geom_bounds_arrays(gen)
gn = np.asarray(H.rand_feasible_geom_norm(gen, lb, ub), float)
barriers = H.build_barriers(gen, gn, lb, ub)
design.add_rotor()
for b in barriers:
    design.add_rotor_barrier(b)
P.set_speed(design, 50.0)
r = design.compute(0.9, 0.9, 0.0, 0.0, NUM_CORES=1)
print("R IS NONE" if r is None else "keys: " + ",".join(sorted(r.keys())), flush=True)
mm = (r or {}).get("means", {})
print("means keys: " + ",".join(sorted(mm.keys())), flush=True)
for k in ("Flux_d1", "Flux_q1", "Flux_d3", "Flux_q3"):
    ok = k in mm
    print("  means[%-9s] %s  %s" % (k, "OK" if ok else "MISSING", (float(mm[k]) if ok else "")), flush=True)
design.delete_rotor(); design.close_project()
