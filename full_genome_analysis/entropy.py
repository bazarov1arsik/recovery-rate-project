import pandas as pd
import numpy as np
from collections import defaultdict
from recovery_core import get_gbr_tsi

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')

def compute_entropy(df, ws):
    window_inds = defaultdict(set)
    for (ind, chrom), grp in df.groupby(['individual', 'CHROM']):
        for s, e in zip(grp['Start'].values // ws, grp['End'].values // ws):
            for b in range(s, e + 1):
                window_inds[(chrom, b)].add(ind)
    counts = np.array([len(v) for v in window_inds.values()])
    if len(counts) == 0:
        return 0.0, 0.0
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    return H, np.exp(H)

for ws in [100_000, 500_000, 1_000_000]:
    H_gbr, Neff_gbr = compute_entropy(gbr_m, ws)
    H_tsi, Neff_tsi = compute_entropy(tsi_m, ws)
    print(f"Окно {ws//1000} kb: GBR H={H_gbr:.3f}, Neff={Neff_gbr:.1f}; TSI H={H_tsi:.3f}, Neff={Neff_tsi:.1f}")