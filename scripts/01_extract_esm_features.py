"""
Extract ESM2 (pretrained protein language model) per-position features for the
Cas9 WT sequence, using a single forward pass. Because every mutant in the
dataset is a single substitution of the SAME wild-type sequence, we do not
need to re-embed each of the 8117 mutants: the wild-type marginal approach
(Meier et al. 2021, "Language models enable zero-shot prediction of the
effects of mutations on protein function") lets us score every substitution
from one WT forward pass.

Outputs (solution/data/):
  - wt_hidden.npy         (L, H) last-layer hidden states for the WT sequence
  - wt_logits.npy         (L, V) vocab logits for the WT sequence (masked-marginal
                             for a masked position is included per-position -- see below)
  - wt_log_probs_plain.npy (L, V) log softmax of plain (unmasked) forward pass
  - vocab.json            token -> id mapping
"""
import json
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import sys
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "facebook/esm2_t12_35M_UR50D"
FASTA = "../CAS9_STRP1_WT.fasta"
OUT_DIR = "data"
# tag output files by model so multiple checkpoints can coexist
TAG = MODEL_NAME.split("/")[-1]

def read_fasta(path):
    with open(path) as f:
        lines = f.read().splitlines()
    seq = "".join(l.strip() for l in lines if not l.startswith(">"))
    return seq

def main():
    seq = read_fasta(FASTA)
    L = len(seq)
    print(f"WT length: {L}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    model.eval()

    enc = tok(seq, return_tensors="pt")
    input_ids = enc["input_ids"]  # includes BOS/EOS
    print("token count (with special tokens):", input_ids.shape[1])

    t0 = time.time()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    print(f"plain forward pass: {time.time()-t0:.1f}s")

    hidden = out.hidden_states[-1][0]  # (L+2, H)
    logits = out.logits[0]  # (L+2, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    # strip BOS/EOS -> positions 1..L correspond to residues 1..L
    hidden = hidden[1:1 + L].numpy()
    log_probs_plain = log_probs[1:1 + L].numpy()

    np.save(f"{OUT_DIR}/wt_hidden_{TAG}.npy", hidden.astype(np.float32))
    np.save(f"{OUT_DIR}/wt_log_probs_plain_{TAG}.npy", log_probs_plain.astype(np.float32))
    with open(f"{OUT_DIR}/vocab_{TAG}.json", "w") as f:
        json.dump(tok.get_vocab(), f)
    with open(f"{OUT_DIR}/wt_seq.txt", "w") as f:
        f.write(seq)
    # keep un-tagged copies pointing at the default/first model for backward compat
    if not os.path.exists(f"{OUT_DIR}/wt_hidden.npy"):
        np.save(f"{OUT_DIR}/wt_hidden.npy", hidden.astype(np.float32))
        np.save(f"{OUT_DIR}/wt_log_probs_plain.npy", log_probs_plain.astype(np.float32))
        with open(f"{OUT_DIR}/vocab.json", "w") as f:
            json.dump(tok.get_vocab(), f)

    print("Saved hidden:", hidden.shape, "log_probs:", log_probs_plain.shape)

if __name__ == "__main__":
    main()
