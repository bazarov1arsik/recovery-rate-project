import pandas as pd
import numpy as np
import glob

MASK_PATH = '20140520.strict_mask.autosomes.bed'

def load_mask(path):
    intervals = {}
    total = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            chrom = parts[0].replace('chr', '')
            start, end = int(parts[1]), int(parts[2])
            if chrom not in intervals:
                intervals[chrom] = []
            intervals[chrom].append((start, end))
            total += end - start
    for chrom in intervals:
        intervals[chrom].sort()
    return intervals, total

def load_population(pop_label, data_dir='.'):
    files = sorted(glob.glob(f"{data_dir}/{pop_label}.YRI.grch37.chr*.em.tsv"))
    if not files:
        raise FileNotFoundError(f"No data found for {pop_label} in {data_dir}")
    dfs = [pd.read_csv(f, sep='\t') for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['CHROM'] = df_all['CHROM'].astype(str)
    df_all['individual'] = df_all['Sample'].str.rsplit('_', n=1).str[0]
    return df_all

def get_gbr_tsi(data_dir='.'):
    gbr = merge_segments(load_population('GBR', data_dir))
    tsi = merge_segments(load_population('TSI', data_dir))
    return gbr, tsi

def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_ivals = sorted(intervals)
    merged = [list(sorted_ivals[0])]
    for start, end in sorted_ivals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

def intersect_with_mask(segments, mask_chrom):
    result = []
    i, j = 0, 0
    segments = sorted(segments)
    mask_chrom = sorted(mask_chrom)
    while i < len(segments) and j < len(mask_chrom):
        lo = max(segments[i][0], mask_chrom[j][0])
        hi = min(segments[i][1], mask_chrom[j][1])
        if lo < hi:
            result.append((lo, hi))
        if segments[i][1] < mask_chrom[j][1]:
            i += 1
        else:
            j += 1
    return result

def merge_segments(df):
    merged = []
    for (ind, chrom), grp in df.groupby(['individual', 'CHROM']):
        intervals = list(zip(grp['Start'], grp['End']))
        for s, e in merge_intervals(intervals):
            merged.append({'individual': ind, 'CHROM': chrom, 'Start': s, 'End': e, 'Length': e - s})
    return pd.DataFrame(merged)

def population_union(df, mask):
    total = 0
    for chrom, grp in df.groupby('CHROM'):
        if chrom not in mask:
            continue
        intervals = list(zip(grp['Start'], grp['End']))
        merged = merge_intervals(intervals)
        clipped = intersect_with_mask(merged, mask[chrom])
        total += sum(e - s for s, e in clipped)
    return total

def compute_R_P(df_all_merged, individuals, G_valid, mask):
    subset = df_all_merged[df_all_merged['individual'].isin(individuals)]
    if subset.empty:
        return 0.0
    return population_union(subset, mask) / G_valid