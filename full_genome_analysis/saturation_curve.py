import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from recovery_core import get_gbr_tsi, merge_intervals, load_mask, MASK_PATH, intersect_with_mask

mask, G_valid = load_mask(MASK_PATH)
print(f"G_valid = {G_valid:,} bp ({G_valid/1e9:.3f} Gb)")

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')

def mask_segments(df, mask):
    rows = []
    for chrom, grp in df.groupby('CHROM'):
        if chrom not in mask:
            continue
        mask_intervals = mask[chrom]
        for _, row in grp.iterrows():
            seg = (row['Start'], row['End'])
            clipped = intersect_with_mask([seg], mask_intervals)
            for s, e in clipped:
                rows.append({
                    'individual': row['individual'],
                    'CHROM': chrom,
                    'Start': s,
                    'End': e,
                    'Length': e - s
                })
    return pd.DataFrame(rows)

print("   Masking GBR segments...")
gbr_masked = mask_segments(gbr_m, mask)
print("   Masking TSI segments...")
tsi_masked = mask_segments(tsi_m, mask)

gbr_inds = gbr_masked['individual'].unique()
tsi_inds = tsi_masked['individual'].unique()
N_gbr = len(gbr_inds)
N_tsi = len(tsi_inds)

B = 100              
np.random.seed(42)

def compute_curve_fast(pop_df_masked, individuals, N, G_valid):
    curves = np.zeros((B, N))
    for b in range(B):
        order = np.random.choice(individuals, size=N, replace=False)
        union_by_chrom = {}
        for n in range(N):
            ind_df = pop_df_masked[pop_df_masked['individual'] == order[n]]
            for chrom, grp in ind_df.groupby('CHROM'):
                if chrom not in union_by_chrom:
                    union_by_chrom[chrom] = []
                union_by_chrom[chrom].extend(zip(grp['Start'], grp['End']))
            total_bp = 0
            for chrom, intervals in union_by_chrom.items():
                if not intervals:
                    continue
                merged = merge_intervals(intervals)
                merged = [(s, e) for s, e in merged]
                union_by_chrom[chrom] = merged
                total_bp += sum(e - s for s, e in merged)
            curves[b, n] = total_bp / G_valid
        if (b + 1) % 10 == 0:               
            elapsed = time.time() - start
            print(f"  {b+1}/{B} ({elapsed:.1f}s)")
    mean_curve = curves.mean(axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    n95 = np.argmax(mean_curve >= 0.95 * mean_curve[-1]) + 1
    return mean_curve, lower, upper, n95

start = time.time()
print("GBR saturation curve...")
mean_gbr, lo_gbr, hi_gbr, n95_gbr = compute_curve_fast(gbr_masked, gbr_inds, N_gbr, G_valid)
print("TSI saturation curve...")
mean_tsi, lo_tsi, hi_tsi, n95_tsi = compute_curve_fast(tsi_masked, tsi_inds, N_tsi, G_valid)

print(f"GBR: n95 = {n95_gbr} (из {N_gbr})")
print(f"TSI: n95 = {n95_tsi} (из {N_tsi})")

plt.figure(figsize=(10, 6))
x_gbr = np.arange(1, N_gbr+1)
x_tsi = np.arange(1, N_tsi+1)
plt.plot(x_gbr, mean_gbr*100, color='blue', label='GBR')
plt.fill_between(x_gbr, lo_gbr*100, hi_gbr*100, color='blue', alpha=0.2)
plt.plot(x_tsi, mean_tsi*100, color='red', label='TSI')
plt.fill_between(x_tsi, lo_tsi*100, hi_tsi*100, color='red', alpha=0.2)
plt.axvline(n95_gbr, color='blue', linestyle='--', alpha=0.5, label=f'GBR n95={n95_gbr}')
plt.axvline(n95_tsi, color='red', linestyle='--', alpha=0.5, label=f'TSI n95={n95_tsi}')
plt.xlabel('Число индивидов')
plt.ylabel('Recovery rate, %')
plt.title('Кривые насыщения recovery rate (callable-маска)')
plt.legend()
plt.tight_layout()
plt.savefig('saturation_curves.png', dpi=150)
print("График сохранён в saturation_curves.png")