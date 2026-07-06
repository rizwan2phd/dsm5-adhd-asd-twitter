"""
Stage 1 — Zero-Shot Depressive Content Screening
=================================================
Screens raw tweets for depressive relevance using a zero-shot NLI
classifier (cross-encoder/nli-deberta-v3-large), before any symptom-level
classification is applied. Corresponds to Methods: "Stage 1: Zero-Shot
Depressive Content Screening".

Input : CSV with a text column (e.g. 'tweet') and a 'disorder' column
         used to restrict to the ADHD/ASD self-report subset.
Output: same CSV + 'predicted_label' and 'predicted_prob' columns
         (predicted_prob = P("DSM-5 depressive symptom present")).
"""

import torch
import time
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = "predictions_1m.csv"
TEXT_COL   = "tweet"
MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
CANDIDATE_LABELS = ["DSM-5 depressive symptom present", "no DSM-5 depressive symptom"]
LABEL_MAP = {
    "DSM-5 depressive symptom present": "Positive",
    "no DSM-5 depressive symptom":      "Negative",
}
POSITIVE_LABEL = "DSM-5 depressive symptom present"
BATCH_SIZE     = 500          # see note below — 2000 is too large for this model
CHECKPOINT_EVERY = 50         # save partial results every N batches
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df = df[df['disorder'].isin(["Adhd_eng", "Asd_eng"])]
print(df['disorder'].value_counts())
print(f"Loaded {len(df)} rows from '{CSV_PATH}'")
print(f"Columns : {df.columns.tolist()}")
print(f"Shape   : {df.shape}\n")

texts = df[TEXT_COL].fillna("").tolist()
print(f"Samples: {len(texts)}  |  Model: {MODEL_NAME}\n")
print(f"CUDA available: {torch.cuda.is_available()}  |  "
      f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=0,
    torch_dtype=torch.float16,   # halves memory, speeds up on H100
)

pred_labels = []
pred_probs  = []
n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
OUTPUT_PATH = CSV_PATH.replace(".csv", "_fullNLI_probsDebv3zerov2.csv")

start_time = time.time()
pbar = tqdm(range(0, len(texts), BATCH_SIZE), total=n_batches, unit="batch")

for batch_idx, i in enumerate(pbar):
    batch   = texts[i : i + BATCH_SIZE]
    results = classifier(batch, candidate_labels=CANDIDATE_LABELS)
    if isinstance(results, dict):
        results = [results]

    for res in results:
        prob_dict = dict(zip(res["labels"], res["scores"]))
        top_label = res["labels"][0]
        top_prob  = prob_dict[POSITIVE_LABEL]
        pred_labels.append(LABEL_MAP[top_label])
        pred_probs.append(round(top_prob, 4))

    done = min(i + BATCH_SIZE, len(texts))
    elapsed = time.time() - start_time
    rate = done / elapsed  # tweets/sec
    remaining = (len(texts) - done) / rate if rate > 0 else 0
    pbar.set_postfix({
        "tweets": f"{done}/{len(texts)}",
        "rate": f"{rate:.1f}/s",
        "ETA": f"{remaining/60:.1f}min"
    })

    # periodic checkpoint so a crash doesn't lose everything
    if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
        tmp = df.iloc[:len(pred_labels)].copy()
        tmp["predicted_label"] = pred_labels
        tmp["predicted_prob"]  = pred_probs
        tmp.to_csv(OUTPUT_PATH + ".partial", index=False)

total_time = time.time() - start_time
print(f"\nDone in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours).")
print(f"Average rate: {len(texts)/total_time:.1f} tweets/sec\n")

df["predicted_label"] = pred_labels
df["predicted_prob"]  = pred_probs
print(df[[TEXT_COL, "predicted_label", "predicted_prob"]].head(10).to_string())

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to '{OUTPUT_PATH}'")
