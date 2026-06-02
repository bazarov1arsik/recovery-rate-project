import pandas as pd
import numpy as np
from recovery_core import get_gbr_tsi, population_union, load_mask, MASK_PATH

_, G_valid = load_mask(MASK_PATH)
mask, _ = load_mask(MASK_PATH)

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')

gbr_inds = gbr_m['individual'].unique()
tsi_inds = tsi_m['individual'].unique()
N_gbr = len(gbr_inds)
N_tsi = len(tsi_inds)

B = 1000
diffs = np.zeros(B)
np.random.seed(42)

for i in range(B):
    gbr_sample = np.random.choice(gbr_inds, size=N_gbr, replace=True)
    tsi_sample = np.random.choice(tsi_inds, size=N_tsi, replace=True)
    R_gbr = population_union(gbr_m[gbr_m['individual'].isin(gbr_sample)], mask) / G_valid
    R_tsi = population_union(tsi_m[tsi_m['individual'].isin(tsi_sample)], mask) / G_valid
    diffs[i] = R_tsi - R_gbr
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{B}")

ci_lo = np.percentile(diffs, 2.5)
ci_hi = np.percentile(diffs, 97.5)
avg_diff = np.mean(diffs)

print(f"Средняя разность: {avg_diff*100:.4f}%")
print(f"95% доверительный интервал: [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]")