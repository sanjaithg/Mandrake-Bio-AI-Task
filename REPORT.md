# Predicting Cas9 Editing Activity — Evaluation of Proposal & Experiment

## 1. The proposal, evaluated

**Proposal:** combine a pretrained protein sequence model with a diffusion-based structure-prediction model, replace the structure trunk with an activity-prediction head, and predict gene-editor activity.

**Verdict: modify, don't pursue as stated.** Keep the pretrained sequence model. Do not adopt the diffusion-structure component without evidence that structural features add value beyond a static crystal structure — evidence this experiment specifically looked for and did not find (Section 6). The proposal also silently assumes the target label is "editing activity"; Section 2 shows the data does not cleanly support that.

## 2. What the measurements support, and the main limitation

`CAS9_STRP1_Spencer_2017_positive`: 8,117 single-point substitution mutants of full-length *S. pyogenes* Cas9 (1368 aa), scored by **flow-cytometry positive selection** — survival/enrichment requires both correct **expression/folding/stability** and **catalytic cleavage activity**. `DMS_score` is a single scalar that conflates these two biologically distinct effects with no way to separate them post hoc.

Consequences:
- This is **not** a clean activity assay; it's closer to "does this variant work as a functional, expressed, catalytically competent protein at all." A destabilizing mutation and a catalytic-dead mutation look identical in the score.
- Single protein, single species, single assay, **single mutants only** — any claim of generalization to other Cas9 orthologs, other nucleases, or combinatorial variants is unsupported by this dataset alone.
- No replicate/uncertainty information per mutant, so measurement noise can't be assessed directly.

**What we can legitimately claim to predict:** relative loss-of-function of full-length SpCas9 single substitutions, as this positive-selection assay defines it. Not "diffusion-relevant catalytic activity" specifically, not cross-species generalization, not combinatorial effects.

## 3. Evaluation protocol

Two held-out splits, 70/15/15 (train/val/test), to make an implicit assumption explicit and testable:

- **Random split** (mutant-level): every residue position is seen in training; only the specific substitution is held out. This is the easy, optimistic regime — closest to interpolation.
- **Position split**: held-out **positions** never appear in training at all (all mutants at that position go together). This is the regime a real deployment scenario needs — predicting effects of mutations at positions with no training label — and it's the only split that can distinguish "the model learned generalizable sequence context" from "the model memorized this position's typical effect."

**Baselines:** (a) train-mean predictor, (b) BLOSUM62 substitution score (classical, no learning), (c) ESM2 wild-type-marginal zero-shot score (Meier et al. 2021; log P(mut)/P(wt) from one forward pass over the WT sequence — no supervised training at all).

**Metric:** Spearman rho between predicted and true `DMS_score` on held-out data — appropriate because DMS scores are only meaningfully ordinal/relative, and this matches how variant-effect predictors are conventionally evaluated (ProteinGym itself uses Spearman).

## 4. Implementation

**Predictor of the supplied scores:** Ridge regression on `[ESM2-t12-35M per-position last-hidden-state (480-d) at the mutated residue] + [one-hot(mutant amino acid)]`. Because every mutant is a single substitution of the *same* WT sequence, only **one ESM2 forward pass** is needed for the whole dataset (wild-type-marginal trick) — this made the whole pipeline run on CPU in under a minute, no GPU required.

**Substantive extension (tests: does structural context help?):** append two features derived from a real solved Cas9 structure, **not a diffusion-generated one** — PDB 4UN3 (SpCas9–sgRNA–target DNA ternary complex, Anders et al. 2014): (1) CA contact number within 10 Å (burial proxy), (2) minimum distance to any bound nucleic-acid atom (proxy for "is this residue near the business end of the enzyme"). 95.5% of residues resolved in the crystal structure; the rest (disordered loops) imputed with the column mean.

*Why a real structure and not an actual diffusion model:* no GPU was available in this environment and the 5-hour budget does not allow standing up an AlphaFold/diffusion structure pipeline from scratch. Using a real crystal structure isolates the *scientific* question — "does structural context help at all?" — from the *engineering* question of which structure-generation method to use. If structural context doesn't help even with a real, high-quality structure, that is stronger evidence against the proposal's structure-diffusion component than any failure of an untested diffusion model would be.

**Ablation:** structure-features-only (no ESM embedding) vs. embedding-only vs. embedding+structure.

**Challenge experiment (designed to attack the preferred interpretation):** a model using **only** a one-hot of the mutated *position* (+ mutant-identity one-hot) — i.e., no pretrained representation, no sequence context, just "memorize this position's typical effect." By construction this feature is uninformative for positions never seen in training, so it isolates how much of any model's apparent skill is genuine transfer vs. memorized per-position statistics.

## 5. Architecture specification

**Executed (this experiment):**
- **Sequence trunk:** ESM2 (HuggingFace `transformers`) — `esm2_t12_35M_UR50D` (35M, headline), plus `esm2_t30_150M_UR50D` (150M) and `esm2_t33_650M_UR50D` (650M) in the tuning pass. Input: WT sequence (1368 aa), BOS/EOS-tokenized. Output used: `hidden_states[-1]` → `(1368, H)` per-residue (H=480/640/1280) after dropping special tokens; `logits` → `(1368, 33)` vocab distribution, for the zero-shot score.
- **Structure component (substitute for the diffusion trunk):** no learned module — geometric features from static PDB 4UN3 coordinates (CA contact count within 10 Å; nearest-atom distance to bound RNA/DNA via Biopython `NeighborSearch`). Output: `(1368, 2)`.
- **Head:** scikit-learn `Ridge` (linear) on `concat[ESM2 hidden @ mutated position (480 or 640-d), one-hot(mutant AA) (20-d), structure features (2-d)]` → scalar `DMS_score` prediction. Training objective: L2-regularized MSE, `alpha` selected on the validation split (grid 1–1000). Tuning pass also tried Kernel Ridge (RBF) and Gradient Boosting as the head.
- **Frozen/retained:** all ESM2 weights (inference-only, no fine-tuning). **Replaced:** the diffusion structure trunk, by static crystal-structure features (§4: isolates whether structural *information* helps before investing in a structure-*generation* module). **Trained:** only the small ridge/GBR/MLP head (≤~500 params) on 5,660–5,681 examples — fine-tuning a 35M–650M trunk on ~6k labels would overfit and was out of budget.

**Proposed but not executed** (needs GPU + more time; kept separate from tested claims, per the assignment's proposal/evidence distinction): replace the static-PDB features with a diffusion structure model's per-residue trunk embedding (e.g. RFdiffusion/ESMFold-style), pooled to the mutated residue and concatenated with the ESM2 vector exactly as §4's executed extension does, feeding the same ridge/GBR head — so any lift over the executed baseline is attributable to the structure module. Training objective unchanged (supervised MSE); the structure trunk would stay frozen, mirroring the executed setup for a fair comparison.

## 6. Results

| Model | Random split ρ | Position split ρ |
|---|---|---|
| Mean baseline (no training) | — (undefined, constant) | — (undefined, constant) |
| BLOSUM62 (no training) | 0.064 | 0.057 |
| ESM2 zero-shot (no training) | 0.056 | 0.056 |
| Position-identity only (no pretraining) | **0.140** | **0.003** |
| Structure-only (no ESM embedding) | 0.144 | 0.043 |
| ESM2 embedding (ridge) | 0.151 | **0.152** |
| ESM2 embedding + structure (ridge) | 0.155 | 0.145 |

(`figures/results_summary.png`; full numbers in `results/metrics.json`; held-out predictions in `results/held_out_predictions_position_split.csv`.)

**Calibration caveat (RMSE):** despite the positive rank correlation above, the ridge model's **RMSE is worse than the trivial train-mean predictor** on both splits (random: 0.612 vs. 0.607; position: 0.635 vs. 0.609 — `results/metrics.json`). So the model captures a weak but real *rank* signal (which mutations are relatively better/worse) without reliable *absolute-magnitude* accuracy. Any downstream use should treat outputs as a ranking, not a calibrated activity estimate — this is a second, independent piece of evidence for going no further than "modify" on the proposal until the label noise problem (Section 2) is addressed.

**Reading the results:**
1. **All effects are modest** (ρ ≈ 0.15 at best). This assay is noisy and conflates expression with activity (Section 2); no model here explains most of the variance.
2. **The challenge experiment worked as intended and changed the interpretation.** In the random split, "position-identity only" (ρ=0.140) is nearly indistinguishable from the full ESM2 model (ρ=0.151) — meaning most of the apparent skill in an easy random split is just memorizing per-position averages, *not* evidence of transferable sequence understanding. Anyone evaluating only with a random split would over-claim.
3. **The position split is the discriminating test.** There, position-identity collapses to ρ≈0.003 (as it must, by construction) and structure-only nearly collapses (0.145→0.043), while the ESM2 embedding model *holds* at ρ=0.152. This is the actual evidence that the pretrained sequence representation carries generalizable signal about unseen positions, beyond memorization.
4. **The structural extension does not help, and mildly hurts under the harder, more relevant split** (0.152 → 0.145 embedding vs. embedding+structure, position split). Two cheap, real-structure-derived features add no measurable value once genuine sequence generalization is already captured by the embedding.

## 7. Recommendation

- **Keep:** the pretrained protein language model half of the proposal. It is the only component shown to generalize to unseen positions in this experiment, and it's cheap (CPU, one forward pass).
- **Reject/defer:** the diffusion-structure trunk as proposed. This experiment did not find evidence that structural context — tested here via features from a real, high-resolution Cas9 crystal structure — improves prediction over the sequence model alone. Since even ground-truth structural features gave no lift, there is no reason to expect a diffusion-*predicted* structure (strictly noisier, and expensive to compute) to do better; the burden of proof for adding that machinery is not met.
- **What remains untested:** (a) whether a genuinely different structural signal (e.g. per-position dynamics/flexibility, DNA-contact geometry from a catalytically engaged rather than static structure, or actual diffusion-model embeddings/confidence) would help — we tested a static crystal structure with two hand-picked features, not the diffusion architecture itself; (b) generalization beyond SpCas9 to other editors — untestable from this single-protein dataset; (c) whether an expression/activity-decoupled label (if it existed) would show a different, cleaner result. (Whether a larger pretrained backbone helps *was* tested — see Section 8 — and the answer is no, it makes the harder split worse.)
- **Next experiment that could change this recommendation:** repeat the position-split comparison with a genuinely dynamic structural signal (e.g., a diffusion-model-derived per-residue representation, or an MD/flexibility proxy) rather than a static crystal structure — the current experiment rules out one specific implementation of "structural context helps," not the general hypothesis, and a positive result there would flip the recommendation toward pursuing the structure component.

## 8. Tuning pass — bigger models, biophysics features, MLP head, domain breakdown

`scripts/05_tune_and_improve.py` swept: (a) three ESM2 checkpoints (35M, 150M,
650M), (b) four biophysical substitution deltas (hydrophobicity, volume,
polarity, charge), (c) Ridge / Kernel Ridge / Gradient Boosting / MLP heads,
validation-selected. `scripts/06_domain_breakdown.py` reruns the headline
(35M) model's position-split predictions grouped by SpCas9 domain.

| Model | Random ρ (test) | Position ρ (test) |
|---|---|---|
| Original (35M ESM2, ridge) | 0.151 | **0.152** |
| 150M, best-of-sweep (embed only / +biophys+struct) | 0.187 / 0.179 | 0.137 / 0.136 |
| 650M, best-of-sweep (embed only / +biophys+struct) | 0.177 / **0.189** | 0.105 / 0.104 |

**Bigger backbones and richer features improve the easy random split but
make the harder position split monotonically worse** (0.152→0.137→0.105 as
the backbone grows 35M→150M→650M). The MLP head never won a single sweep
cell (Kernel Ridge/GBR always did), so this isn't a missing-nonlinearity
problem — more capacity buys more ability to fit per-position idiosyncrasies,
which the random split rewards and the position split penalizes, the same
memorization gap the Section 6 challenge experiment targets. This answers a
question Section 7 left open: a larger backbone does not raise the
generalization ceiling, it lowers it.

**Per-domain (position-split, 35M ridge model, `results/domain_breakdown.json`,
`figures/domain_breakdown.png`):** RuvC ρ=−0.07, Bridge/other ρ=−0.04, REC
ρ=**0.14**, HNH ρ=**0.22**, PI ρ=0.03. The pooled ρ=0.152 is carried almost
entirely by HNH (catalytic) and REC (recognition lobe); the model is
at-or-below zero in RuvC and near-zero in PI. Per-domain n is small
(7–87 held-out positions), so treat this as a follow-up lead, not a
confirmed effect.

## 9. Reproduction

```
cd cas9_activity_prediction && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python scripts/01_extract_esm_features.py   # 35M model
python scripts/02_structural_features.py    # PDB 4UN3
python scripts/03_build_dataset.py && python scripts/04_run_experiments.py && python scripts/06_domain_breakdown.py
python scripts/01_extract_esm_features.py facebook/esm2_t30_150M_UR50D && python scripts/05_tune_and_improve.py esm2_t30_150M_UR50D
python scripts/01_extract_esm_features.py facebook/esm2_t33_650M_UR50D && python scripts/05_tune_and_improve.py esm2_t33_650M_UR50D
```
