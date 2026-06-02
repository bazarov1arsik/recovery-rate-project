import pandas as pd
import glob
import sys
from recovery_core import load_mask, load_population, merge_intervals, intersect_with_mask, population_union, MASK_PATH

def mean_admixture(df, mask, G_valid):
    ind_admixture = []
    for ind, ind_df in df.groupby('individual'):
        ind_bp = 0
        for chrom, grp in ind_df.groupby('CHROM'):
            if chrom not in mask:
                continue
            intervals = list(zip(grp['Start'], grp['End']))
            merged = merge_intervals(intervals)
            clipped = intersect_with_mask(merged, mask[chrom])
            ind_bp += sum(e - s for s, e in clipped)
        ind_admixture.append(ind_bp / G_valid)
    return sum(ind_admixture) / len(ind_admixture)

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    mask, G_valid = load_mask(MASK_PATH)
    print(f'G_valid: {G_valid:,} bp ({G_valid/1e9:.3f} Gb)')

    gbr = load_population('GBR', data_dir)
    tsi = load_population('TSI', data_dir)

    R_gbr = population_union(gbr, mask) / G_valid
    R_tsi = population_union(tsi, mask) / G_valid
    A_gbr = mean_admixture(gbr, mask, G_valid)
    A_tsi = mean_admixture(tsi, mask, G_valid)
    DA = abs(A_gbr - A_tsi) / ((A_gbr + A_tsi) / 2)

    print(f'GBR: R_P = {R_gbr*100:.4f}%,  A_P = {A_gbr*100:.4f}%')
    print(f'TSI: R_P = {R_tsi*100:.4f}%,  A_P = {A_tsi*100:.4f}%')
    print(f'ΔR = {(R_tsi - R_gbr)*100:.4f}%')
    print(f'DA = {DA*100:.2f}%')