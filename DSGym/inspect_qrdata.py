from dsgym.datasets import DatasetRegistry

dataset = DatasetRegistry.load("qrdata")
samples = dataset.load(limit=2)

print(f"=== {len(samples)} samples loaded ===\n")

def safe_preview(v, n=400):
    try:
        s = repr(v)
    except Exception as e:
        s = f"<unrepr-able: {e}>"
    return s[:n]

for i, s in enumerate(samples):
    print(f"--- Sample {i} ---")
    print(f"Top-level keys: {list(s.keys())}")
    for k, v in s.items():
        print(f"  {k}: ({type(v).__name__}) {safe_preview(v)}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"      {kk}: ({type(vv).__name__}) {safe_preview(vv, 300)}")
    print()