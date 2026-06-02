from recovery_core import get_gbr_tsi
gbr_m, _ = get_gbr_tsi('.')
print("Максимальная позиция:", gbr_m['End'].max())