import pandas as pd
import numpy as np
import sys
from recovery_core import get_gbr_tsi, merge_intervals, population_union, load_mask, MASK_PATH

DEFAULT_MASK_PATH = MASK_PATH
MASK_PATH_ARG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MASK_PATH

def null_model(df, mask, G_valid, B=1000):
    np.random.seed(42)
    null_rates = []
    for b in range(B):
        placed_segments = []
        for chrom, grp in df.groupby('CHROM'):
            if chrom not in mask:
                continue
            intervals = list(zip(grp['Start'], grp['End']))
            segments = [(e - s, s, e) for s, e in intervals]
            segments.sort(reverse=True)
            placed = []
            for length, _, _ in segments:
                max_attempts = 100
                placed_flag = False
                for _ in range(max_attempts):
                    idx = np.random.randint(0, len(mask[chrom]))
                    m_start, m_end = mask[chrom][idx]
                    if m_end - m_start >= length:
                        start = np.random.randint(m_start, m_end - length + 1)
                        end = start + length
                        overlap = False
                        for p_start, p_end in placed:
                            if start < p_end and end > p_start:
                                overlap = True
                                break
                        if not overlap:
                            placed.append((start, end))
                            placed_segments.append((start, end))
                            placed_flag = True
                            break
                if not placed_flag:
                    placed_segments.append((None, None))
        merged = merge_intervals([seg for seg in placed_segments if seg[0] is not None])
        null_rates.append(sum(e - s for s, e in merged) / G_valid)
    return np.mean(null_rates), np.std(null_rates), null_rates

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')

mask, G_valid = load_mask(MASK_PATH_ARG)

for name, df in [("GBR", gbr_m), ("TSI", tsi_m)]:
    R_obs = population_union(df, mask) / G_valid
    mean_null, std_null, _ = null_model(df, mask, G_valid)
    Z = (R_obs - mean_null) / std_null if std_null > 0 else 0
    print(f"{name}: R_obs = {R_obs*100:.2f}%, mean_null = {mean_null*100:.2f}%, Z = {Z:.2f}")