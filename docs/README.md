# `newparam` line — documentation

Documentation home for the **new SynRM rotor parameterisation** line of research
(branch `newparam`). This line replaces the three Hackl-style parameterisations
with one unified, high-dimensional **`RadialSpline`** description (D=114) intended
as the substrate for high-dimensional / latent-space Bayesian optimisation.

This is a **separate line** from the sister PFN/Gibbs-prior work in the
`../MachineDesign/` worktree — see `../HANDOFF.md`. Do not cross-pollinate.
The PFN line's paper lives in its own Overleaf submodule (`../../overleaf`);
this line will get its **own** Overleaf project when there is a paper to write
(a *new* submodule, never a subfolder of the PFN paper — see the rationale in
the session log / `../HANDOFF.md`).

## Reading order

1. [PARAMETERISATION.md](PARAMETERISATION.md) — the design spec and **decisions
   ledger (P1–P8)**. Source of truth for the parameterisation. Start here.
2. [METHODS.md](METHODS.md) — surveyed HD-BO / structured-space BO methods,
   ranked by suitability, with **externally-verified** citations.
3. [experiments/](experiments/) — gate results, figures, and logs (geometry
   gate, latent gate).

Decisions are kept **inline in `PARAMETERISATION.md` §1** (single source of
truth — no separate `DECISIONS.md`, to avoid drift).

---

## Governing hygiene policies

These apply to everything under this line. They extend the parent rules in
`../../CLAUDE.md` (§11 data hygiene; the global literature/citation rules).

### H-CITE — every citation is externally verified

No reference enters `METHODS.md` (or any future bibliography / whitepaper) until
its **authors, title, venue, year** have been read from an **authoritative
source** — Crossref (`api.crossref.org/works/{DOI}`), the arXiv API/abs page, or
DBLP. **Never** take citation metadata from search-engine snippets, Google
Scholar result lists, or LLM summaries (the documented main source of
hallucinated author names).

- Verified entries carry a ✅ and their arXiv id / DOI.
- Unverified candidates live in a clearly separated "to verify" list and are
  **not** cited as fact.
- A reference suggested by a reviewer or assistant is a **hypothesis** until
  verified — verify before adopting.
- PDFs of papers we actually use are archived under [`refs/`](refs/) with the
  `{firstauthor}{year}_{shorttitle}.pdf` convention; paywalled items are queued
  in `refs/paywalled.md`.

### H-REPL — replication validates on the method's own elementary setup first

Before any surveyed method is trusted on **our** problem (RadialSpline + FEA),
we first reproduce it on the **elementary / canonical setup from its own paper**
(its toy function, its reported benchmark, its released code where available)
and confirm we recover the paper's qualitative result. Only then do we apply it
to RadialSpline.

Rationale: this separates "the method works and we wired it correctly" from
"the method does/doesn't suit our problem" — the same discipline that turned the
sister line's multi-week detour into a finding (`../../CLAUDE.md` §13). Each
replication records: the elementary setup used, the paper's claimed result, our
reproduced result, and the verdict, under [`experiments/`](experiments/).
