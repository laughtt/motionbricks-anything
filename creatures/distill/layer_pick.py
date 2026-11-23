"""Step 1: pick which pose-transformer layers to keep, by angular distance between hidden states.

Idea (depth pruning): a layer that changes the hidden state very little can be dropped. We measure
d(i, j) = mean angular distance between the hidden state before layer i and the hidden state before layer j
(index 0 = the input of layer 0, index L = the output of the last layer), on masked teacher requests.
The greedy pick then removes, one by one, the layer whose removal leaves the smallest gap
d(prev_out, next_in). Layers 0, 1 and the last layer are protected.

Two ways to use this file:
  * distill.py select       runs the teacher on requests and writes <work>/runs/angular.json (GPU + NVIDIA checkpoint)
  * layer_pick.py --table angular.json --keep 10       prints the greedy keep list from a saved table (CPU only)
The functions below (accumulate_angular, pick_keep, best_block) are pure maths and work on any hidden states.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def accumulate_angular(H, valid):
    """H: torch tensor [L+1, B, P, D] of hidden states (input of layer 0 + output of every layer);
    valid: float tensor [B, P] (1 for real positions, 0 for padding).
    Returns (sum_ij [L+1, L+1] tensor, count): the angular distance (0..1 = angle / pi) summed over valid positions."""
    import torch
    import torch.nn.functional as Fn
    Hn = Fn.normalize(H, dim=-1)
    cos = torch.einsum("ibpd,jbpd->ijbp", Hn, Hn).clamp(-1, 1)
    ang = torch.arccos(cos) / math.pi
    return (ang * valid).sum((-1, -2)), valid.sum()


def pick_keep(D, n_keep, protect=(0, 1, 15)):
    """Greedy: repeatedly remove the layer whose removal leaves the smallest gap, measured as the angular distance
    between the hidden state the removed layer would read and the one the next kept layer reads.
    D: [L+1, L+1] distance table. Returns the sorted list of kept layer indices."""
    D = np.asarray(D)
    L = D.shape[0] - 1
    keep = list(range(L))
    while len(keep) > n_keep:
        best, bi = 9, None
        for k, li in enumerate(keep):
            if li in protect:
                continue
            prev_out = keep[k - 1] + 1 if k > 0 else 0
            nxt_in = keep[k + 1] if k + 1 < len(keep) else L
            c = D[prev_out, nxt_in]
            if c < best:
                best, bi = c, k
        keep.pop(bi)
    return keep


def best_block(D, n_remove):
    """Best contiguous block of n_remove layers to drop (layer 0 is kept). Returns (distance, first removed layer)."""
    D = np.asarray(D)
    L = D.shape[0] - 1
    return min((float(D[i, i + n_remove]), i) for i in range(1, L - n_remove + 1))


def main():
    ap = argparse.ArgumentParser(description="Print the greedy keep list from an angular-distance table made by 'distill.py select'.")
    ap.add_argument("--table", required=True, help="angular.json written by distill.py select")
    ap.add_argument("--keep", type=int, default=10, help="number of layers to keep (default 10 of 16)")
    a = ap.parse_args()
    D = np.array(json.loads(Path(a.table).read_text())["D"])
    print(f"keep {a.keep}: {pick_keep(D, a.keep)}")


if __name__ == "__main__":
    main()
