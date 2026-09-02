from loader import get_dataloaders

for run in (1, 2):
    tr, te, va = get_dataloaders(16, augmented=False, vit_transformed=True)
    y = [float(v) for _, yb in te for v in yb]
    print(f"run{run}: train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
    print(f"       first5_test_bmi = {[round(v,2) for v in y[:5]]}")
