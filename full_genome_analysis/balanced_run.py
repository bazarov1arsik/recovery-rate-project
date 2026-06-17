import pandas as pd
import numpy as np
import sys
from recovery_core import load_population, merge_segments, population_union, get_gbr_tsi, load_mask, MASK_PATH
import time

G_valid = 2_064_554_803
N_TARGET = 91  
RANDOM_SEED = 42

mask, _ = load_mask(MASK_PATH)
G_valid = sum(e - s for chrom in mask for s, e in mask[chrom])

print("Загрузка GBR и TSI...")
gbr_raw = load_population('GBR', '.')
tsi_raw = load_population('TSI', '.')

gbr_m = merge_segments(gbr_raw)
tsi_m = merge_segments(tsi_raw)

gbr_inds = gbr_m['individual'].unique()
tsi_inds_all = tsi_m['individual'].unique()

print(f"Исходные размеры: GBR {len(gbr_inds)}, TSI {len(tsi_inds_all)}")

np.random.seed(RANDOM_SEED)
tsi_inds_balanced = np.random.choice(tsi_inds_all, size=N_TARGET, replace=False)

print(f"Отобрано TSI: {len(tsi_inds_balanced)} индивидов")

def compute_R_A(df, individuals, mask):
    R = population_union(df[df['individual'].isin(individuals)], mask) / G_valid
    A = df[df['individual'].isin(individuals)].groupby('individual')['Length'].sum().mean() / G_valid
    return R, A

R_gbr, A_gbr = compute_R_A(gbr_m, gbr_inds, mask)
R_tsi, A_tsi = compute_R_A(tsi_m, tsi_inds_balanced, mask)

DA = abs(A_gbr - A_tsi) / ((A_gbr + A_tsi) / 2)
delta = R_tsi - R_gbr

print(f"\nGBR: R = {R_gbr*100:.4f}%, A = {A_gbr*100:.4f}%")
print(f"TSI (balanced): R = {R_tsi*100:.4f}%, A = {A_tsi*100:.4f}%")
print(f"ΔR = {delta*100:.4f}%, DA = {DA*100:.2f}%")

B_BOOT = 1000
diffs_boot = np.zeros(B_BOOT)
np.random.seed(RANDOM_SEED)

print("\nБутстрап...")
for i in range(B_BOOT):
    gbr_sample = np.random.choice(gbr_inds, size=len(gbr_inds), replace=True)
    tsi_sample = np.random.choice(tsi_inds_balanced, size=len(tsi_inds_balanced), replace=True)
    Rg = population_union(gbr_m[gbr_m['individual'].isin(gbr_sample)], mask) / G_valid
    Rt = population_union(tsi_m[tsi_m['individual'].isin(tsi_sample)], mask) / G_valid
    diffs_boot[i] = Rt - Rg
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{B_BOOT}")

ci_lo = np.percentile(diffs_boot, 2.5)
ci_hi = np.percentile(diffs_boot, 97.5)
print(f"95% CI: [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]")

B_PERM = 10000
all_individuals = list(gbr_inds) + list(tsi_inds_balanced)
N1 = len(gbr_inds)
N2 = len(tsi_inds_balanced)
exceed = 0

print(f"\nПерестановочный тест (B={B_PERM})...")
start = time.time()
for i in range(B_PERM):
    perm = np.random.permutation(all_individuals)
    pseudo_gbr = perm[:N1]
    pseudo_tsi = perm[N1:]
    Rg = population_union(gbr_m[gbr_m['individual'].isin(pseudo_gbr)], mask) / G_valid
    Rt = population_union(tsi_m[tsi_m['individual'].isin(pseudo_tsi)], mask) / G_valid
    if abs(Rt - Rg) >= abs(delta):
        exceed += 1
    if (i+1) % 1000 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{B_PERM} ({elapsed:.0f}s)")

p_value = (exceed + 1) / (B_PERM + 1)
print(f"p-value = {p_value:.4f}")

with open('balanced_results.txt', 'w') as f:
    f.write(f"GBR (N={len(gbr_inds)}): R = {R_gbr*100:.4f}%, A = {A_gbr*100:.4f}%\n")
    f.write(f"TSI balanced (N={len(tsi_inds_balanced)}): R = {R_tsi*100:.4f}%, A = {A_tsi*100:.4f}%\n")
    f.write(f"ΔR = {delta*100:.4f}%\n")
    f.write(f"DA = {DA*100:.2f}%\n")
    f.write(f"Bootstrap 95% CI: [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]\n")
    f.write(f"Permutation test p-value = {p_value:.4f}\n")

print("\nРезультаты сохранены в balanced_results.txt")