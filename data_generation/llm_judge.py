import json, re
from tqdm import tqdm
import ollama

MODEL_NAME = "qwen2.5:32b"

# ---------------- constants ----------------
FORBIDDEN_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "no"
}
MEDICAL_CONTEXT_WORDS = {
    "no", "not", "without", "possible", "likely", "unlikely",
    "suggests", "appears", "may", "might"
}
VALID_LABELS = {"ANAT-DP", "OBS-DP", "OBS-DA", "OBS-U"}

# ---------------- helpers -------------------
def debug_filtering(word, label, report, reason=""):
    print(f"REMOVED: '{word}' ({label}) - {reason}")

def prefilter_labels(report, labels):
    """Strip invalid / stop-word pairs before they reach the LLM."""
    clean = {}
    for w, l in labels.items():
        if not isinstance(l, str) or l.strip() not in VALID_LABELS:
            debug_filtering(w, l, report, "Invalid label (pre-filter)")
            continue
        if w.lower() in FORBIDDEN_WORDS and w.lower() not in MEDICAL_CONTEXT_WORDS:
            debug_filtering(w, l, report, "Stop-word (pre-filter)")
            continue
        clean[w] = l.strip()
    return clean

# ------------- prompt builder --------------
def build_prompt(report, labels_dict):
    """Load the LLM Judge prompt from a text file and insert the report + labels."""
    prompt_path = "/Users/saishashetty/Desktop/RadAnnotate/data_generation/prompts/llm_judge.txt"
    with open(prompt_path, "r") as f:
        template = f.read()
    # Escape braces used for formatting (to preserve JSON braces in examples)
    template = template.replace("{", "{{").replace("}", "}}")
    # Restore the two intended placeholders
    template = template.replace("{{report}}", "{report}").replace("{{labels_dict}}", "{labels_dict}")
    return template.format(
        report=report,
        labels_dict=json.dumps(labels_dict, indent=2)
    )

# -------------- LLM extractor --------------
def extract_labels_dict(text):
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "labels" in parsed:
            return parsed["labels"]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for pat in [
        r'"labels"\s*:\s*(\{{[^}}]+\}})',
        r'"labels"\s*:\s*(\{{[\s\S]+?\}})\s*\}}',
        r'\{{[\s\S]*"labels"\s*:\s*(\{{[\s\S]+?\}})\s*\}}',
    ]:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return {}

# ---------- post-validation logic ----------
def validate_and_clean_labels(report, pre_labels, llm_labels):
    final, removed, restored = {}, 0, 0
    print(f"\nProcessing report: {report[:50]}...")
    print(f"Pre-filtered labels: {len(pre_labels)}")
    print(f"LLM returned:       {len(llm_labels)}")

    # Accept anything the LLM returned that survives local checks
    for w, l in llm_labels.items():
        if not isinstance(l, str) or l.strip() not in VALID_LABELS:
            debug_filtering(w, l, report, "Invalid label (LLM)")
            removed += 1
            continue
        if w.lower() in FORBIDDEN_WORDS and w.lower() not in MEDICAL_CONTEXT_WORDS:
            debug_filtering(w, l, report, "Stop-word (LLM)")
            removed += 1
            continue
        final[w] = l.strip()

    # Restore only if LLM mentioned but was filtered
    for w, l in pre_labels.items():
        if w in llm_labels and w not in final:
            final[w] = l
            restored += 1
            print(f"Restored: '{w}' ({l}) – was in LLM output but filtered")

    print(f"Removed:  {removed}")
    print(f"Restored: {restored}")
    print(f"Final labels: {len(final)}")
    return final

# --------------- main loop -----------------
def validate_labels_llm(input_path):
    with open(input_path) as f:
        data = json.load(f)

    cleaned, base_out = [], input_path.replace(".json", "v2_llm_cleaned.json")
    checkpoint = base_out.replace(".json", "_checkpoint.json")

    for i, item in tqdm(enumerate(data), total=len(data), desc="Validating (1 call per report)"):
        report = item.get("Report", "")
        orig_labels = item.get("Labels", {}) or item.get("labels", {})

        print(f"\n{'='*60}\nProcessing item {i+1}/{len(data)}")

        if not report.strip():
            print("Empty report, skipping")
            cleaned.append({"Report": report, "Labels": {}})
            continue

        pre_labels = prefilter_labels(report, orig_labels)
        prompt = build_prompt(report, pre_labels)

        try:
            resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            llm_labels = extract_labels_dict(resp["message"]["content"])
            final_labels = validate_and_clean_labels(report, pre_labels, llm_labels)
        except Exception as e:
            print(f"LLM error: {e}")
            final_labels = pre_labels
            print(f"Using pre-filtered labels ({len(final_labels)})")

        cleaned.append({"Report": report, "Labels": final_labels})

        if (i + 1) % 10 == 0:
            with open(checkpoint, "w") as fck:
                json.dump(cleaned, fck, indent=2)
            print(f"Checkpoint saved ({len(cleaned)} / {len(data)})")

    with open(base_out, "w") as fout:
        json.dump(cleaned, fout, indent=2)
    print(f"Final cleaned file saved to: {base_out}")

# --------------- run ------------------------
if __name__ == "__main__":
    validate_labels_llm("/home/spshetty/RadAnnotate/data_generation/data/combinedv1.json")
