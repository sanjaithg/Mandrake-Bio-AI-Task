"""
Main experiment runner.

Splits
------
- random_split: 70/15/15 random split over mutants. Tests interpolation
  within seen positions (easiest, most optimistic regime).
- position_split: split by RESIDUE POSITION (70/15/15 of the 1368 positions,
  all mutants at a held-out position go together). Tests generalization to
  UNSEEN positions -- the generalization claim that actually matters for
  "will this predict a mutation we haven't scored," and the one a real
  variant-effect predictor needs to support.

Models
------
- mean_baseline: predicts the training mean (sanity floor)
- blosum62: substitution-matrix score b(wt,mut), no learning
- esm_zero_shot: ESM2 wild-type-marginal log-odds, no supervised training
- ridge_embedding: Ridge regression on ESM2 per-position embedding (480-d)
  + one-hot(mut_aa) -- "sequence model" supervised predictor (the proposal's
  sequence-model half)
- ridge_embedding_struct: same + 2 structural features from a real Cas9
  crystal structure (contact number, distance to bound nucleic acid) --
  the "structural context" extension. NOTE: this is a real solved structure,
  not a diffusion-generated one; see report for why we did not run an actual
  diffusion structure model under the time/compute budget.

Outputs: results/metrics.json, results/held_out_predictions_position_split.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

RNG = 0
DATA_DIR = "data"
RESULTS_DIR = "results"

AAS = list("ACDEFGHIKLMNPQRSTVWY")

# BLOSUM62 (standard 20x20 log-odds matrix)
from Bio.Align import substitution_matrices
BLOSUM62 = substitution_matrices.load("BLOSUM62")

def load():
    d = np.load(f"{DATA_DIR}/features.npz", allow_pickle=True)
    return d

def make_splits(d, mode, seed=RNG):
    n = len(d["mutant"])
    if mode == "random":
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    elif mode == "position":
        positions = d["position"]
        uniq = np.unique(positions)
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq)
        n_train = int(0.7 * len(uniq))
        n_val = int(0.15 * len(uniq))
        train_pos = set(uniq[:n_train])
        val_pos = set(uniq[n_train:n_train + n_val])
        test_pos = set(uniq[n_train + n_val:])
        train_idx = np.where(np.isin(positions, list(train_pos)))[0]
        val_idx = np.where(np.isin(positions, list(val_pos)))[0]
        test_idx = np.where(np.isin(positions, list(test_pos)))[0]
        return train_idx, val_idx, test_idx
    else:
        raise ValueError(mode)

def onehot_mut(mut_aa_arr):
    idx_map = {a: i for i, a in enumerate(AAS)}
    oh = np.zeros((len(mut_aa_arr), len(AAS)), dtype=np.float32)
    for i, a in enumerate(mut_aa_arr):
        oh[i, idx_map[a]] = 1.0
    return oh

def onehot_wt(wt_aa_arr):
    return onehot_mut(wt_aa_arr)

def blosum_score(wt_aa_arr, mut_aa_arr):
    out = np.zeros(len(wt_aa_arr), dtype=np.float32)
    for i, (w, m) in enumerate(zip(wt_aa_arr, mut_aa_arr)):
        out[i] = BLOSUM62[w, m]
    return out

def spearman(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return float(r)

def fit_ridge(Xtr, ytr, Xval, yval, alphas=(0.1, 1, 3, 10, 30, 100, 300)):
    best_alpha, best_rho, best_model = None, -2, None
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)
    for a in alphas:
        m = Ridge(alpha=a, random_state=RNG).fit(Xtr_s, ytr)
        rho = spearman(yval, m.predict(Xval_s))
        if rho > best_rho:
            best_alpha, best_rho, best_model = a, rho, m
    return best_model, scaler, best_alpha, best_rho

def main():
    d = load()
    y = d["DMS_score"]
    n = len(y)
    emb = d["esm_embedding"]
    oh = onehot_mut(d["mut_aa"])
    oh_wt = onehot_wt(d["wt_aa"])
    struct = np.stack([d["struct_contact_number"], d["struct_min_dist_nucleic"]], axis=1)
    blosum = blosum_score(d["wt_aa"], d["mut_aa"])
    zero_shot = d["esm_zero_shot"]

    X_base = np.concatenate([emb, oh], axis=1)
    X_ext = np.concatenate([emb, oh, struct], axis=1)
    # CHALLENGE FEATURE SET: no pretrained representation at all -- just a
    # one-hot of the mutated POSITION (memorizes a per-position mean/offset,
    # no sequence context) plus one-hot(mut_aa). This can only "generalize"
    # to a held-out position by chance, so it isolates how much of
    # ridge_embedding's skill is genuine transfer from pretraining vs.
    # memorizing per-position statistics that happen to be seen in training.
    positions = d["position"]
    uniq_positions = np.unique(positions)
    pos_index = {p: i for i, p in enumerate(uniq_positions)}
    oh_pos = np.zeros((len(positions), len(uniq_positions)), dtype=np.float32)
    for i, p in enumerate(positions):
        oh_pos[i, pos_index[p]] = 1.0
    X_identity_only = np.concatenate([oh_pos, oh], axis=1)

    metrics = {}
    held_out_records = {}

    for split_mode in ["random", "position"]:
        train_idx, val_idx, test_idx = make_splits(d, split_mode)
        print(f"\n=== split={split_mode} === train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        split_metrics = {}

        # --- zero-training baselines (evaluated on test only, no fitting) ---
        # A constant predictor has no rank correlation (Spearman is undefined
        # for a constant vector), so report RMSE instead -- the metric a
        # constant predictor is actually well-defined on. This is the error
        # floor every other model's RMSE should beat.
        mean_pred = np.full(len(test_idx), y[train_idx].mean())
        split_metrics["mean_baseline_spearman"] = None  # undefined by construction
        split_metrics["mean_baseline_rmse"] = float(np.sqrt(np.mean((y[test_idx] - mean_pred) ** 2)))
        split_metrics["blosum62"] = spearman(y[test_idx], blosum[test_idx])
        split_metrics["esm_zero_shot"] = spearman(y[test_idx], zero_shot[test_idx])

        # --- supervised: embedding only ---
        model, scaler, alpha, val_rho = fit_ridge(X_base[train_idx], y[train_idx], X_base[val_idx], y[val_idx])
        pred_test = model.predict(scaler.transform(X_base[test_idx]))
        split_metrics["ridge_embedding"] = spearman(y[test_idx], pred_test)
        split_metrics["ridge_embedding_rmse"] = float(np.sqrt(np.mean((y[test_idx] - pred_test) ** 2)))
        split_metrics["ridge_embedding_alpha"] = alpha
        preds_embedding = pred_test.copy()

        # --- supervised: embedding + structure (extension) ---
        model2, scaler2, alpha2, val_rho2 = fit_ridge(X_ext[train_idx], y[train_idx], X_ext[val_idx], y[val_idx])
        pred_test2 = model2.predict(scaler2.transform(X_ext[test_idx]))
        split_metrics["ridge_embedding_struct"] = spearman(y[test_idx], pred_test2)
        split_metrics["ridge_embedding_struct_alpha"] = alpha2

        # --- ablation: structure-only (no ESM embedding at all) ---
        X_struct_only = np.concatenate([struct, oh], axis=1)
        model3, scaler3, alpha3, _ = fit_ridge(X_struct_only[train_idx], y[train_idx], X_struct_only[val_idx], y[val_idx])
        pred_test3 = model3.predict(scaler3.transform(X_struct_only[test_idx]))
        split_metrics["ridge_struct_only"] = spearman(y[test_idx], pred_test3)

        # --- challenge experiment: position-identity-only (no pretrained rep) ---
        model4, scaler4, alpha4, _ = fit_ridge(X_identity_only[train_idx], y[train_idx], X_identity_only[val_idx], y[val_idx])
        pred_test4 = model4.predict(scaler4.transform(X_identity_only[test_idx]))
        split_metrics["ridge_position_identity_only"] = spearman(y[test_idx], pred_test4)

        for k, v in split_metrics.items():
            print(f"  {k}: {v}")
        metrics[split_mode] = split_metrics

        if split_mode == "position":
            out_df = pd.DataFrame({
                "mutant": d["mutant"][test_idx],
                "position": d["position"][test_idx],
                "DMS_score_true": y[test_idx],
                "pred_ridge_embedding": preds_embedding,
                "pred_ridge_embedding_struct": pred_test2,
                "pred_esm_zero_shot": zero_shot[test_idx],
                "pred_blosum62": blosum[test_idx],
            })
            out_df.to_csv(f"{RESULTS_DIR}/held_out_predictions_position_split.csv", index=False)

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved results/metrics.json and results/held_out_predictions_position_split.csv")

if __name__ == "__main__":
    main()
