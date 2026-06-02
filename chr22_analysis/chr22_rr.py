import pandas as pd
import sys
from recovery_core import merge_intervals, intersect_with_mask

def load_mask(mask_file):
    intervals = {}
    with open(mask_file) as f:
        for line in f:
            parts = line.strip().split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            if chrom not in intervals:
                intervals[chrom] = []
            intervals[chrom].append((start, end))
    for chrom in intervals:
        intervals[chrom].sort()
    return intervals

def recovery_rate_chr22(em_file, mask_file):
    mask = load_mask(mask_file)
    gvalid = 0
    for ivals in mask.values():
        for s, e in ivals:
            gvalid += e - s

    df = pd.read_csv(em_file, sep='\t')
    df['individual'] = df['Sample'].str.rsplit('_', n=1).str[0]

    pop_segments = {}
    for chrom, grp in df.groupby('CHROM'):
        chrom = str(chrom)
        pop_segments[chrom] = list(zip(grp['Start'], grp['End']))

    unique_bp = 0
    for chrom, segs in pop_segments.items():
        if chrom not in mask:
            continue
        merged = merge_intervals(segs)
        clipped = intersect_with_mask(merged, mask[chrom])
        unique_bp += sum(e - s for s, e in clipped)

    return unique_bp, unique_bp / gvalid * 100

if __name__ == '__main__':
    bases, rate = recovery_rate_chr22(sys.argv[1], sys.argv[2])
    print(f'Уникальное покрытие: {bases} п.н.')
    print(f'Recovery Rate: {rate:.4f}%')