# Cas9 activity prediction — solution


## Layout

- `scripts/01_extract_esm_features.py` — one ESM2 forward pass over the Cas9 WT
  sequence (pretrained model, per-position embeddings + logits).
- `scripts/02_structural_features.py` — real Cas9 crystal structure (PDB 4UN3)
  → per-residue contact number + distance-to-nucleic-acid features.
- `scripts/03_build_dataset.py` — parses all 8117 mutants, attaches ESM2
  zero-shot score, embedding, structural features → `data/features.npz`.
- `scripts/04_run_experiments.py` — baselines, supervised ridge models,
  ablation, and the challenge experiment, across random and position splits
  → `results/metrics.json`, `results/held_out_predictions_position_split.csv`.
- `scripts/05_tune_and_improve.py` — tuning pass: takes an ESM2 checkpoint tag
  as an argument (150M and 650M were both tried), added biophysical
  substitution features, and a Ridge/Kernel-Ridge/Gradient-Boosting/MLP sweep
  → `results/metrics_tuned_<tag>.json` (see `REPORT.md` §8 for why the
  original, smaller-model numbers remain the headline result).
- `scripts/06_domain_breakdown.py` — splits the position-split held-out
  predictions by SpCas9 domain (RuvC/REC/HNH/PI) and recomputes ρ per domain
  → `results/domain_breakdown.json`, `figures/domain_breakdown.png` (see
  `REPORT.md` §9).
- `scripts/make_pdf.py` — renders `REPORT.md` to PDF (uses system Python +
  `reportlab`, not the venv).
- `figures/results_summary.png` — bar chart of all models × both splits.
- `figures/domain_breakdown.png` — per-domain ρ breakdown.
- `figures/REPORT.pdf` — rendered PDF of the report.

## Reproduce

Run from inside `cas9_activity_prediction/` (paths in the scripts are relative
to this directory):

```bash
cd cas9_activity_prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/01_extract_esm_features.py
python scripts/02_structural_features.py
python scripts/03_build_dataset.py
python scripts/04_run_experiments.py
python scripts/06_domain_breakdown.py

# optional tuning pass (slower: ~5 min per checkpoint, fits a bigger model + a small sweep)
python scripts/01_extract_esm_features.py facebook/esm2_t30_150M_UR50D
python scripts/05_tune_and_improve.py esm2_t30_150M_UR50D
python scripts/01_extract_esm_features.py facebook/esm2_t33_650M_UR50D
python scripts/05_tune_and_improve.py esm2_t33_650M_UR50D

# optional: re-render the PDF (uses system python3 + reportlab, not the venv)
python3 scripts/make_pdf.py REPORT.md figures/REPORT.pdf "Cas9 Activity Prediction — Report" --compact
```

Core pipeline (steps 01–04, 06) runs end-to-end on CPU in well under 5 minutes
(no GPU required). Internet access needed once, to download the ESM2 weights
(facebook/esm2_t12_35M_UR50D, ~150MB via HuggingFace) and the PDB structure
(4UN3, via RCSB). The tuning pass additionally downloads the 150M-parameter
(~600MB) and 650M-parameter (~2.5GB) ESM2 checkpoints and takes a few minutes
longer per checkpoint (kernel-ridge/gradient-boosting/MLP fitting).
