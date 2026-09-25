"""
Build structure-derived per-residue features from a real Cas9 crystal structure
(PDB 4UN3: SpCas9 + sgRNA + target DNA ternary complex, Anders et al. 2014,
Nature). This is NOT a diffusion structure prediction model -- it is a real
solved structure used as a cheap, CPU-only proxy for "structural context" so
we can test whether structural information helps, without claiming anything
about diffusion-model features we never ran (see report).

Features per residue (author numbering, chain A = Cas9):
  - contact_number: # of other protein CA atoms within 10A (burial proxy)
  - min_dist_nucleic: min distance (any atom) to sgRNA or target DNA chains
                       (proxy for "is this residue near the business end of
                       the enzyme, i.e. nucleic-acid-binding/catalytic region")

We align these to the WT sequence by matching PDB SEQRES/author numbering
against our 1-indexed sequence positions (Cas9 numbering in this dataset is
the standard SpCas9 numbering and matches 4UN3 author numbering directly for
the ordered residues; PDB has ~gaps for disordered loops, which we impute
with the mean feature value).
"""
import os
import numpy as np
import requests
from Bio.PDB import PDBParser, NeighborSearch, is_aa

PDB_ID = "4UN3"
PDB_PATH = "data/4UN3.pdb"
WT_SEQ_PATH = "data/wt_seq.txt"
OUT_DIR = "data"

PROTEIN_CHAIN = "B"           # 4UN3 chain B = Cas9 protein
NUCLEIC_CHAINS = ["A", "C", "D"]  # A = sgRNA, C/D = target/non-target DNA strands

def fetch_pdb_if_missing():
    if os.path.exists(PDB_PATH):
        return
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PDB_PATH, "wb") as f:
        f.write(r.content)
    print(f"Saved {PDB_PATH} ({len(r.content)} bytes)")

def main():
    fetch_pdb_if_missing()
    wt_seq = open(WT_SEQ_PATH).read().strip()
    L = len(wt_seq)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("4UN3", PDB_PATH)
    model = structure[0]

    protein_atoms = []
    residue_by_resnum = {}
    for chain in model:
        if chain.id != PROTEIN_CHAIN:
            continue
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            resnum = res.id[1]
            residue_by_resnum[resnum] = res
            if "CA" in res:
                protein_atoms.append(res["CA"])

    nucleic_atoms = []
    for chain in model:
        if chain.id in NUCLEIC_CHAINS:
            for res in chain:
                for atom in res:
                    nucleic_atoms.append(atom)

    print(f"Protein CA atoms: {len(protein_atoms)}, nucleic atoms: {len(nucleic_atoms)}")
    ns_protein = NeighborSearch(protein_atoms)
    ns_nucleic = NeighborSearch(nucleic_atoms) if nucleic_atoms else None

    contact_number = np.full(L, np.nan, dtype=np.float32)
    min_dist_nucleic = np.full(L, np.nan, dtype=np.float32)

    for resnum, res in residue_by_resnum.items():
        pos = resnum - 1  # convert to 0-indexed sequence position
        if pos < 0 or pos >= L:
            continue
        if "CA" not in res:
            continue
        ca = res["CA"]
        neighbors = ns_protein.search(ca.coord, 10.0)
        contact_number[pos] = len(neighbors) - 1  # exclude self

        if ns_nucleic is not None:
            dmin = min(np.linalg.norm(a.coord - ca.coord) for a in
                       ns_nucleic.search(ca.coord, 40.0)) if ns_nucleic.search(ca.coord, 40.0) else 40.0
            min_dist_nucleic[pos] = dmin

    n_resolved = np.sum(~np.isnan(contact_number))
    print(f"Resolved structural features for {n_resolved}/{L} residues "
          f"({100*n_resolved/L:.1f}%); rest imputed with column mean "
          f"(disordered loops / crystal construct gaps).")

    for arr in (contact_number, min_dist_nucleic):
        mean_val = np.nanmean(arr)
        arr[np.isnan(arr)] = mean_val

    np.save(f"{OUT_DIR}/struct_contact_number.npy", contact_number)
    np.save(f"{OUT_DIR}/struct_min_dist_nucleic.npy", min_dist_nucleic)

if __name__ == "__main__":
    main()
