"""
Tuning pass: try to beat the script-04 baseline results with (a) a larger
pretrained ESM2 checkpoint, (b) cheap biophysical substitution features
(hydrophobicity/charge/volume/polarity deltas -- classical DMS-predictor
features, essentially free to compute), and (c) a small model-family sweep
(Ridge / Kernel Ridge (RBF) / Gradient Boosting / MLP) with validation-selected
hyperparameters, all evaluated on the same random vs. position splits so
results are directly comparable to results/metrics.json.

Everything here is still ESM2 (sequence model) + optional structure/biophysics
features -- no diffusion model. This is a hyperparameter/feature tuning pass,
not a new architecture.

Usage: python scripts/05_tune_and_improve.py [ESM2_TAG]
  ESM2_TAG must match the tag used in step 01's output filenames, e.g.
  esm2_t30_150M_UR50D or esm2_t33_650M_UR50D. Defaults to the 150M checkpoint.
"""
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from Bio.Align import substitution_matrices

RNG = 0
DATA_DIR = "data"
RESULTS_DIR = "results"
CSV_PATH = "../CAS9_STRP1_Spencer_2017_positive.csv"
TAG = sys.argv[1] if len(sys.argv) > 1 else "esm2_t30_150M_UR50D"

AAS = list("ACDEFGHIKLMNPQRSTVWY")
BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Classical amino-acid scales (Kyte-Doolittle hydropathy, Zimmerman volume A^3,
# Grantham polarity, net charge at physiological pH). Cheap, interpretable,
# standard in DMS effect-prediction literature.
HYDROPATHY = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,
                   I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
VOLUME = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,
              I=166.7,L=166.7,K=168.6,M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)
POLARITY = dict(A=8.1,R=10.5,N=11.6,D=13.0,C=5.5,Q=10.5,E=12.3,G=9.0,H=10.4,
                 I=5.2,L=4.9,K=11.3,M=5.7,F=5.2,P=8.0,S=9.2,T=8.6,W=5.4,Y=6.2,V=5.9)
CHARGE = dict(D=-1,E=-1,K=1,R=1,H=0.1)

def parse_mutant(code):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", code)
    return m.group(1), int(m.group(2)), m.group(3)

def onehot(arr):
    idx_map = {a: i for i, a in enumerate(AAS)}
    oh = np.zeros((len(arr), len(AAS)), dtype=np.float32)
    for i, a in enumerate(arr):
        oh[i, idx_map[a]] = 1.0
    return oh

def biophys_deltas(wt_aa, mut_aa):
    n = len(wt_aa)
    out = np.zeros((n, 4), dtype=np.float32)
    for i, (w, m) in enumerate(zip(wt_aa, mut_aa)):
        out[i, 0] = HYDROPATHY[m] - HYDROPATHY[w]
        out[i, 1] = VOLUME[m] - VOLUME[w]
        out[i, 2] = POLARITY[m] - POLARITY[w]
        out[i, 3] = CHARGE.get(m, 0) - CHARGE.get(w, 0)
    return out

def spearman(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return float(r)

def make_splits(positions, mode, seed=RNG):
    n = len(positions)
    if mode == "random":
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_train, n_val = int(0.7 * n), int(0.15 * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    uniq = np.unique(positions)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_train, n_val = int(0.7 * len(uniq)), int(0.15 * len(uniq))
    train_pos, val_pos, test_pos = set(uniq[:n_train]), set(uniq[n_train:n_train+n_val]), set(uniq[n_train+n_val:])
    return (np.where(np.isin(positions, list(train_pos)))[0],
            np.where(np.isin(positions, list(val_pos)))[0],
            np.where(np.isin(positions, list(test_pos)))[0])

MAX_KRR_TRAIN = 1500  # Kernel Ridge is O(n^3) to fit exactly; cap n for tractability

def fit_best(Xtr, ytr, Xval, yval):
    """Sweep Ridge alpha, Kernel Ridge (RBF) gamma/alpha (on a subsample --
    exact KRR is O(n^3), infeasible at n~5000 on CPU), and Gradient Boosting
    depth/n_estimators; return the best-on-validation model."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)

    best = dict(rho=-2, name=None, model=None)

    for a in [1, 3, 10, 30, 100, 300, 1000]:
        m = Ridge(alpha=a, random_state=RNG).fit(Xtr_s, ytr)
        rho = spearman(yval, m.predict(Xval_s))
        print(f"    ridge(alpha={a}): val_rho={rho:.4f}")
        if rho > best["rho"]:
            best = dict(rho=rho, name=f"ridge(alpha={a})", model=m)

    if len(Xtr_s) > MAX_KRR_TRAIN:
        rng = np.random.RandomState(RNG)
        sub = rng.choice(len(Xtr_s), MAX_KRR_TRAIN, replace=False)
        Xtr_krr, ytr_krr = Xtr_s[sub], ytr[sub]
    else:
        Xtr_krr, ytr_krr = Xtr_s, ytr
    for a in [1, 10]:
        for g in [0.001, 0.01]:
            m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xtr_krr, ytr_krr)
            rho = spearman(yval, m.predict(Xval_s))
            print(f"    kernel_ridge(alpha={a},gamma={g}, n_train={len(Xtr_krr)}): val_rho={rho:.4f}")
            if rho > best["rho"]:
                best = dict(rho=rho, name=f"kernel_ridge(alpha={a},gamma={g},subsampled_to={len(Xtr_krr)})", model=m)

    for depth in [2, 3]:
        for n_est in [100]:
            m = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=RNG).fit(Xtr_s, ytr)
            rho = spearman(yval, m.predict(Xval_s))
            print(f"    gbr(depth={depth},n_est={n_est}): val_rho={rho:.4f}")
            if rho > best["rho"]:
                best = dict(rho=rho, name=f"gbr(depth={depth},n_est={n_est})", model=m)

    # Non-linear head with a hidden layer -- tests whether the ridge/GBR/KRR
    # sweep above was missing signal a small neural net could pick up.
    # early_stopping=True holds out 10% of Xtr internally and stops once
    # validation loss stalls, since this is the one model family here prone
    # to overfitting ~5-6k training rows.
    for hidden in [(64,), (128,), (128, 32)]:
        for alpha in [1e-3, 1e-2, 1e-1]:
            m = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                              activation="relu", solver="adam",
                              early_stopping=True, n_iter_no_change=15,
                              max_iter=500, random_state=RNG).fit(Xtr_s, ytr)
            rho = spearman(yval, m.predict(Xval_s))
            print(f"    mlp(hidden={hidden},alpha={alpha}): val_rho={rho:.4f}")
            if rho > best["rho"]:
                best = dict(rho=rho, name=f"mlp(hidden={hidden},alpha={alpha})", model=m)

    return best, scaler

def main():
    df = pd.read_csv(CSV_PATH, usecols=["mutant", "DMS_score"])
    wt_seq = open(f"{DATA_DIR}/wt_seq.txt").read().strip()
    hidden = np.load(f"{DATA_DIR}/wt_hidden_{TAG}.npy")
    log_probs = np.load(f"{DATA_DIR}/wt_log_probs_plain_{TAG}.npy")
    vocab = json.load(open(f"{DATA_DIR}/vocab_{TAG}.json"))
    contact = np.load(f"{DATA_DIR}/struct_contact_number.npy")
    min_dist_na = np.load(f"{DATA_DIR}/struct_min_dist_nucleic.npy")

    wt_aas, positions, mut_aas = [], [], []
    for code in df["mutant"]:
        w, p, m = parse_mutant(code)
        wt_aas.append(w); positions.append(p); mut_aas.append(m)
    positions = np.array(positions)
    wt_aas, mut_aas = np.array(wt_aas), np.array(mut_aas)
    idx0 = positions - 1

    y = df["DMS_score"].values.astype(np.float32)
    emb = hidden[idx0]
    zero_shot = np.array([log_probs[i, vocab[m]] - log_probs[i, vocab[w]]
                           for i, w, m in zip(idx0, wt_aas, mut_aas)], dtype=np.float32)
    oh_mut = onehot(mut_aas)
    struct = np.stack([contact[idx0], min_dist_na[idx0]], axis=1)
    biophys = biophys_deltas(wt_aas, mut_aas)

    print(f"Larger-model zero-shot sanity check: rho={spearman(y, zero_shot):.3f} "
          f"(script-04 with 35M model got 0.056/0.056)")

    # Tuned feature set: bigger embedding + mut one-hot + structure + biophysics
    X_tuned = np.concatenate([emb, oh_mut, struct, biophys], axis=1)
    # For comparison: tuned feature set minus the new bits (bigger embedding only)
    X_bigger_emb_only = np.concatenate([emb, oh_mut], axis=1)

    results = {}
    for split_mode in ["random", "position"]:
        train_idx, val_idx, test_idx = make_splits(positions, split_mode)
        print(f"\n=== split={split_mode} ===")

        split_res = {}
        for name, X in [("bigger_embedding_only", X_bigger_emb_only),
                         ("tuned_full", X_tuned)]:
            best, scaler = fit_best(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            pred_test = best["model"].predict(scaler.transform(X[test_idx]))
            rho_test = spearman(y[test_idx], pred_test)
            rmse_test = float(np.sqrt(np.mean((y[test_idx] - pred_test) ** 2)))
            print(f"  {name}: best_model={best['name']} val_rho={best['rho']:.3f} "
                  f"TEST_rho={rho_test:.3f} TEST_rmse={rmse_test:.3f}")
            split_res[name] = dict(best_model=best["name"], val_rho=best["rho"],
                                    test_rho=rho_test, test_rmse=rmse_test)
        results[split_mode] = split_res

    out_path = f"{RESULTS_DIR}/metrics_tuned_{TAG}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    if TAG == "esm2_t30_150M_UR50D":
        # keep the un-tagged filename the report/README already reference
        with open(f"{RESULTS_DIR}/metrics_tuned.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved results/metrics_tuned.json")

if __name__ == "__main__":
    main()
