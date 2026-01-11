import os, json
from sklearn.model_selection import train_test_split
from data_paths import ENTITY_FILES

OUT_DIR = "outputs/splits"
SEED = 42
TEST_SIZE = 0.5

def dedup_by_report(items):
    seen = set()
    out = []
    for it in items:
        rep = (it.get("report") or "").strip()
        if rep in seen:
            continue
        seen.add(rep)
        out.append(it)
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for ent, path in ENTITY_FILES.items():
        raw = json.load(open(path))
        uniq = dedup_by_report(raw)

        val, test = train_test_split(uniq, test_size=TEST_SIZE, random_state=SEED)

        ent_dir = os.path.join(OUT_DIR, ent)
        os.makedirs(ent_dir, exist_ok=True)

        json.dump(val, open(os.path.join(ent_dir, "val.json"), "w"), indent=2)
        json.dump(test, open(os.path.join(ent_dir, "test.json"), "w"), indent=2)

        print(f"{ent}: raw={len(raw)} → uniq={len(uniq)} | val={len(val)} test={len(test)}")

if __name__ == "__main__":
    main()
