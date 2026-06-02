import pandas as pd
import numpy as np
import time
from recovery_core import get_gbr_tsi, population_union, load_mask, MASK_PATH

_, G_valid = load_mask(MASK_PATH)
mask, _ = load_mask(MASK_PATH)

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')

gbr_m['population'] = 'GBR'
tsi_m['population'] = 'TSI'
all_m = pd.concat([gbr_m, tsi_m], ignore_index=True)

gbr_inds = list(gbr_m['individual'].unique())
tsi_inds = list(tsi_m['individual'].unique())
N_gbr = len(gbr_inds)
N_tsi = len(tsi_inds)

R_gbr_obs = population_union(all_m[all_m['individual'].isin(gbr_inds)], mask) / G_valid
R_tsi_obs = population_union(all_m[all_m['individual'].isin(tsi_inds)], mask) / G_valid
T_obs = R_tsi_obs - R_gbr_obs

all_individuals = gbr_inds + tsi_inds
B = 10_000
np.random.seed(42)

print(f"Наблюдаемая разность: {T_obs*100:.4f}%")
print(f"Запуск перестановок (B={B})...")
start = time.time()
exceed_count = 0

for i in range(B):
    perm = np.random.permutation(all_individuals)
    pseudo_gbr = perm[:N_gbr]
    pseudo_tsi = perm[N_gbr:]
    R_A = population_union(all_m[all_m['individual'].isin(pseudo_gbr)], mask) / G_valid
    R_B = population_union(all_m[all_m['individual'].isin(pseudo_tsi)], mask) / G_valid
    if abs(R_A - R_B) >= abs(T_obs):
        exceed_count += 1
    if (i + 1) % 1000 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{B} ({elapsed:.1f}s)")

p_value = (exceed_count + 1) / (B + 1)
print(f"Перестановок: {B}, превышений: {exceed_count}")
print(f"p-value: {p_value:.4f}")