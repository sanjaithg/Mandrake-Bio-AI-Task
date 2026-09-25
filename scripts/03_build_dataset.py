"""
Parse mutant codes (e.g. A1023D), attach:
  - ESM2 wild-type-marginal score (zero-shot baseline)
  - ESM2 per-position hidden state (WT) as the mutated-position embedding
  - structural features at the mutated position
Saves a single feature table (features.npz) + labels for all downstream scripts.
"""
import json
import re
import numpy as np
import pandas as pd

DATA_DIR = "data"
CSV_PATH = "../CAS9_STRP1_Spencer_2017_positive.csv"

AA3 = None  # not needed, single-letter mutant codes

def parse_mutant(code):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", code)
    if not m:
        raise ValueError(f"bad mutant code: {code}")
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mut_aa

def main():
    df = pd.read_csv(CSV_PATH, usecols=["mutant", "DMS_score", "DMS_score_bin"])
    print(f"Loaded {len(df)} mutants")

    wt_seq = open(f"{DATA_DIR}/wt_seq.txt").read().strip()
    hidden = np.load(f"{DATA_DIR}/wt_hidden.npy")            # (L, 480)
    log_probs = np.load(f"{DATA_DIR}/wt_log_probs_plain.npy")  # (L, 33)
    vocab = json.load(open(f"{DATA_DIR}/vocab.json"))
    contact = np.load(f"{DATA_DIR}/struct_contact_number.npy")
    min_dist_na = np.load(f"{DATA_DIR}/struct_min_dist_nucleic.npy")

    wt_aas, positions, mut_aas = [], [], []
    esm_score = np.zeros(len(df), dtype=np.float32)
    emb = np.zeros((len(df), hidden.shape[1]), dtype=np.float32)
    struct_contact = np.zeros(len(df), dtype=np.float32)
    struct_dist_na = np.zeros(len(df), dtype=np.float32)

    bad = 0
    for i, code in enumerate(df["mutant"]):
        wt_aa, pos, mut_aa = parse_mutant(code)
        idx = pos - 1  # 1-indexed -> 0-indexed
        if idx < 0 or idx >= len(wt_seq) or wt_seq[idx] != wt_aa:
            bad += 1
        wt_aas.append(wt_aa)
        positions.append(pos)
        mut_aas.append(mut_aa)

        esm_score[i] = log_probs[idx, vocab[mut_aa]] - log_probs[idx, vocab[wt_aa]]
        emb[i] = hidden[idx]
        struct_contact[i] = contact[idx]
        struct_dist_na[i] = min_dist_na[idx]

    print(f"WT mismatches vs metadata sequence: {bad} / {len(df)}")

    out = dict(
        mutant=df["mutant"].values,
        position=np.array(positions, dtype=np.int32),
        wt_aa=np.array(wt_aas),
        mut_aa=np.array(mut_aas),
        DMS_score=df["DMS_score"].values.astype(np.float32),
        DMS_score_bin=df["DMS_score_bin"].values.astype(np.int32),
        esm_zero_shot=esm_score,
        esm_embedding=emb,
        struct_contact_number=struct_contact,
        struct_min_dist_nucleic=struct_dist_na,
    )
    np.savez_compressed(f"{DATA_DIR}/features.npz", **out)
    print("Saved data/features.npz")
    print("Zero-shot ESM score vs DMS_score Spearman (sanity check):")
    from scipy.stats import spearmanr
    r, p = spearmanr(esm_score, df["DMS_score"])
    print(f"  rho={r:.3f} (p={p:.1e})")

if __name__ == "__main__":
    main()
