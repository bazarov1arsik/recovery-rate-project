import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from recovery_core import get_gbr_tsi, merge_intervals

window_size = 1_000_000

print(">> Loading and merging GBR/TSI data...")
gbr_m, tsi_m = get_gbr_tsi('.')
print("Максимальная позиция в GBR:", gbr_m['End'].max())
def get_pop_union_df(df):
    rows = []
    for chrom, grp in df.groupby('CHROM'):
        intervals = sorted(zip(grp['Start'], grp['End']))
        merged = merge_intervals(intervals)
        for s, e in merged:
            rows.append({'CHROM': chrom, 'Start': s, 'End': e})
    return pd.DataFrame(rows)

gbr_union = get_pop_union_df(gbr_m)
tsi_union = get_pop_union_df(tsi_m)

def window_coverage_union(df, ws):
    tmp = defaultdict(list)
    for chrom, start, end in df[['CHROM', 'Start', 'End']].values:
        s_idx = start // ws
        e_idx = end // ws
        for b in range(s_idx, e_idx + 1):
            w_s = b * ws
            w_e = w_s + ws
            overlap_start = max(start, w_s)
            overlap_end = min(end, w_e)
            if overlap_end > overlap_start:
                tmp[(chrom, b)].append((overlap_start, overlap_end))
    result = {}
    for key, intervals in tmp.items():
        intervals.sort()
        merged = []
        for a, b in intervals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        total_bp = sum(e - s for s, e in merged)
        result[key] = total_bp
    return result

cov_gbr = window_coverage_union(gbr_union, window_size)
cov_tsi = window_coverage_union(tsi_union, window_size)

all_keys = sorted(set(list(cov_gbr.keys()) + list(cov_tsi.keys())))

D_w = []
for key in all_keys:
    c_gbr = cov_gbr.get(key, 0) / window_size
    c_tsi = cov_tsi.get(key, 0) / window_size
    D_w.append(c_tsi - c_gbr)

plt.figure(figsize=(14, 5))
plt.scatter(range(len(D_w)), D_w, s=2, alpha=0.6)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Окна (упорядочены по хромосомам)')
plt.ylabel('D_w = доля_TSI - доля_GBR')
plt.title('Геномный профиль разности recovery rate (окна 1 Мб)')
plt.tight_layout()
plt.savefig('window_profile.png', dpi=150)
print("График сохранён в window_profile.png")