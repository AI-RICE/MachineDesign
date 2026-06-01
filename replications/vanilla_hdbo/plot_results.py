"""Plot log10-regret vs BO iteration from cached traces in results/.

  .venv/bin/python replications/vanilla_hdbo/plot_results.py --func hartmann6 --dim 25
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
COLORS = {"dsp": "C0", "default": "C3", "random": "0.5"}
LABELS = {"dsp": "DSP (√D LogNormal prior)", "default": "default Γ(3,6) prior", "random": "random search"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="hartmann6")
    ap.add_argument("--dim", type=int, default=25)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    n_init = None
    summary = {}
    for method in ("dsp", "default", "random"):
        files = sorted(glob.glob(os.path.join(OUT, f"{args.func}_d{args.dim}_{method}_s*.npz")))
        if not files:
            continue
        curves = []
        for fp in files:
            d = np.load(fp)
            best = d["best"]
            f_opt = float(d["f_opt"])
            n_init = int(d["n_init"])
            curves.append(np.log10(np.maximum(best - f_opt, 1e-10)))
        L = min(len(c) for c in curves)
        C = np.stack([c[:L] for c in curves])
        x = np.arange(L) - n_init  # iteration 0 = end of Sobol init
        mean, std = C.mean(0), C.std(0)
        ax.plot(x, mean, color=COLORS[method], label=f"{LABELS[method]} (n={len(files)})")
        ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.15)
        summary[method] = (float(mean[-1]), float(std[-1]))
    ax.axvline(0, ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("BO iteration (0 = end of Sobol init)")
    ax.set_ylabel("log₁₀ regret")
    ax.set_title(f"Vanilla-HDBO replication — {args.func} embedded in D={args.dim}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT, f"regret_{args.func}_d{args.dim}.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)
    for m, (mn, sd) in summary.items():
        print(f"  {m:8s} final log10-regret {mn:+.3f} ± {sd:.3f}")


if __name__ == "__main__":
    main()
