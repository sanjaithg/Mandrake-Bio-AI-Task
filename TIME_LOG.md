# Time, compute, and tools record

**AI tools used:** Antigravity, as a coding assistant for boilerplate/syntax
help while writing the scripts. All research and design decisions — reading
the assignment, the linked Spencer et al. 2017 paper, and the ESM2/ProteinGym
documentation, choosing the wild-type-marginal scoring trick to avoid
per-mutant forward passes, the splits, baselines, feature choices, and the
challenge experiment — are my own and are stated explicitly in the report so
they can be challenged or defended live. I have no prior background in
protein modeling, so I spent extra time upfront on the reading above before
writing any code.

**Compute:** CPU only, no GPU available on this machine. ESM2-t12-35M-UR50D
(35M params) chosen specifically to fit a CPU budget. Total feature extraction:
~30s (one forward pass over the 1368-residue WT sequence). Structural feature
computation from PDB 4UN3: a few seconds. Full experiment run (2 splits × 5
models, ridge regression over 8117 examples): well under a minute.

**Wall-clock time spent (approximate):**
- Reading assignment, data pack, background paper, planning approach: ~45 min
- Environment setup (venv, torch/transformers/biopython/sklearn): ~15 min
- ESM2 feature extraction script + wild-type-marginal design: ~30 min
- Structural feature script (PDB fetch, chain ID debugging, contact/distance
  computation, numbering-alignment sanity check): ~40 min
- Dataset assembly + sanity checks: ~20 min
- Experiment runner (splits, baselines, ridge models, ablation, challenge
  experiment) + debugging: ~50 min
- Verification pass (rerunning the pipeline from a clean state, catching a
  reproducibility bug and a calibration/RMSE gap the correlation metric hid): ~30 min
- Plot + report + time log: ~40 min
- Tuning pass (150M ESM2 checkpoint, biophysical features, model sweep): ~30 min
- **Subtotal: ~4.5 hours** (within the original 5-hour limit)

**Follow-up session (outside the original 5-hour window, done to close out
items flagged below as unfinished):**
- Extracted a third, larger ESM2 checkpoint (650M params) and reran the
  tuning sweep on it: ~20 min (mostly one-time download; blocked briefly
  by the machine running low on disk space mid-download, resolved by
  freeing space and retrying).
- Added an MLP head to the Ridge/Kernel-Ridge/Gradient-Boosting sweep and
  reran it on the 150M and 650M checkpoints: ~15 min.
- Built the per-domain (RuvC/HNH/REC/PI) breakdown script and plot: ~25 min.
- Folded all three sets of new results into REPORT.md and regenerated the
  PDF: ~20 min.
- **Follow-up subtotal: ~1.3 hours**

**Unfinished / explicitly out of scope:**
- No diffusion structure model was run (see report §5 for why, and what a
  positive future result would need to show) — still the single largest
  untested piece of the original proposal.
- No larger ESM2 checkpoint beyond 650M (e.g. 3B) was tried; 650M itself
  already showed the position-split ceiling falling with scale (see report
  §8), so a still-larger model seems unlikely to reverse that trend, but
  it is not tested.
- No hyperparameter search beyond the fixed grids in the Ridge/Kernel-Ridge/
  Gradient-Boosting/MLP sweep (e.g. no random/Bayesian search, no deeper MLPs).
- Per-domain breakdown (report §9) used literature-standard, approximate
  domain boundaries, not boundaries re-derived from the structure directly;
  and per-domain sample sizes are small (as few as 7 held-out positions),
  so those numbers are a lead, not a confirmed effect.
- Structural features are limited to two hand-designed scalars (contact number,
  distance to nucleic acid) from one static crystal structure; richer structural
  representations (e.g. full per-residue structure-model embeddings) were not
  attempted.
