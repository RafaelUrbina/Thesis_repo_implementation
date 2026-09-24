import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from tqdm import tqdm

# 1. Load Pretrained Models directly from Hugging Face Hub
seq_path = project_root / "unicausal_finetunning/models/patentbert-unicausal-seq"
SEQ_MODEL = str(seq_path) if seq_path.exists() else "tanfiona/unicausal-seq-baseline"
tok_path = project_root / "unicausal_finetunning/models/patentbert-unicausal-tok"
TOK_MODEL = str(tok_path) if tok_path.exists() else "tanfiona/unicausal-tok-baseline"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_tokenizer = AutoTokenizer.from_pretrained(SEQ_MODEL)
seq_model = AutoModelForSequenceClassification.from_pretrained(SEQ_MODEL).to(device)

tok_tokenizer = AutoTokenizer.from_pretrained(TOK_MODEL)
tok_model = AutoModelForTokenClassification.from_pretrained(TOK_MODEL).to(device)

# Ensure models are set to evaluation mode
seq_model.eval()
tok_model.eval()

print(f"[INFO] Sequence Model ID2Label: {seq_model.config.id2label}")
print(f"[INFO] Token Model ID2Label: {tok_model.config.id2label}")


print(f"[VERIFY] Sequence Model Path/Name: {seq_model.name_or_path}")
print(f"[VERIFY] Token Model Path/Name: {tok_model.name_or_path}")
# 2. Define Prediction Functions
def classify_causality(text: str, threshold: float = 0.5) -> bool:
    text = str(text).strip()
    if not text:
        return False

    inputs = seq_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.inference_mode():
        outputs = seq_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Index 0 is Causal in your PatentBERT sequence model
    causal_prob = probs[0].item()
    return causal_prob >= threshold

def extract_spans(text: str):
    text = str(text).strip()
    if not text:
        return "", ""

    inputs = tok_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = tok_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    cause_chars, effect_chars = [], []

    # Token Map based on your PatentBERT Diagnostic Predictions:
    # 0 = Outside (O)
    # 1, 2 = Cause Spans
    # 4 = Effect Spans
    # (Label 3 represents Signal words like "due to", which are excluded from cause/effect)
    CAUSE_LABEL_IDS = {1, 2}
    EFFECT_LABEL_IDS = {4}

    for idx, pred_id in enumerate(preds):
        start, end = offset_mapping[idx]
        
        # Ignore special tokens ([CLS], [SEP], [PAD])
        if start == end:
            continue

        pred_int = int(pred_id)
        if pred_int in CAUSE_LABEL_IDS:
            cause_chars.append((start, end))
        elif pred_int in EFFECT_LABEL_IDS:
            effect_chars.append((start, end))

    cause_str = (
        text[cause_chars[0][0] : cause_chars[-1][1]] if cause_chars else ""
    )
    effect_str = (
        text[effect_chars[0][0] : effect_chars[-1][1]] if effect_chars else ""
    )

    return cause_str, effect_str

# 3. Execution Pipeline for CSV with Checkpoint/Resume Support
def process_csv(input_path: str, text_column: str, output_path: str, save_every: int = 50, threshold: float = 0.5):
    # If output_path already exists, resume from it!
    target_path = output_path if Path(output_path).exists() else input_path
    
    rows = []
    with open(target_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            fieldnames = [text_column]
        for fn in ["is_causal", "cause_span", "effect_span"]:
            if fn not in fieldnames:
                fieldnames.append(fn)
        for row in reader:
            rows.append(row)

    for idx, row in enumerate(tqdm(rows, desc="Extracting Causal Relations")):
        # Check if already processed (checkpoint resume check)
        existing_causal = row.get("is_causal", "")
        existing_cause_span = row.get("cause_span", "")
        
        if existing_causal != "" and existing_causal is not None:
            is_caus_bool = str(existing_causal).lower() in ["true", "1", "yes"]
            # If it's marked as causal but cause_span is empty, re-process it!
            if not is_caus_bool or existing_cause_span != "":
                continue

        text_str = row.get(text_column, "")
        if not text_str or not str(text_str).strip():
            row["is_causal"] = False
            row["cause_span"] = ""
            row["effect_span"] = ""
        else:
            is_caus = classify_causality(str(text_str), threshold=threshold)
            row["is_causal"] = is_caus
            if is_caus:
                c, e = extract_spans(str(text_str))
                row["cause_span"] = c
                row["effect_span"] = e
            else:
                row["cause_span"] = ""
                row["effect_span"] = ""

        # Incremental save checkpoint every `save_every` rows
        if (idx + 1) % save_every == 0:
            with open(output_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    # Final save
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    input_file = PATENT_PIPELINE_PATH / "unsupervised_link/output/causal_filtered_unsupervised_sentiment_pos.csv"
    output_file = PATENT_PIPELINE_PATH / "unsupervised_link/output/unicausal_sentiment_filtered_unsupervised.csv"
    process_csv(input_path=str(input_file), text_column="text", output_path=str(output_file), threshold=0.5)