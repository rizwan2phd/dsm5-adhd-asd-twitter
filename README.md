# Population-Level Profiling of DSM-5 Depressive Symptoms Among Self-Reported ADHD and ASD Users on Twitter

Code accompanying the paper *"Population-Level Profiling of DSM-5 Depressive
Symptoms Among Self-Reported ADHD and ASD Users on Twitter: An Exploratory
Study Using Advanced NLP and Statistical Analysis."*

This repository implements the full pipeline described in the paper's
Methods section: a two-stage tweet classification pipeline (zero-shot
pre-filter → fine-tuned multi-label DSM-5 symptom classifier), per-label
threshold calibration, and the downstream group-level discriminative and
symptom co-occurrence analyses.

## Pipeline overview

| Step | Script | Paper section |
|---|---|---|
| 1 | `src/01_zero_shot_nli_prefilter.py` | Stage 1: Zero-Shot Depressive Content Screening |
| 2 | `src/02_mentalroberta_gridsearch_multilabel.py` | Stage 2: Multi-Class Model Training / Per-Label Decision Threshold Calibration |
| 3 | `src/03_final_train_and_tweet_inference.py` | Final model training + User Symptom Scoring |
| 4 | `src/04_discriminative_and_cooccurrence_analysis.py` | Group-Level Comparison + Symptom Co-occurrence Analysis |

Run in order. Each script's outputs feed the next:

```
01_zero_shot_nli_prefilter.py
        │  (adds predicted_label, predicted_prob)
        ▼
02_mentalroberta_gridsearch_multilabel.py
        │  (selects best hyperparameters + per-label thresholds
        │   from redsm5_M5.csv; saves best_model/)
        ▼
03_final_train_and_tweet_inference.py
        │  (retrains on all labelled data with best params;
        │   scores the full tweet corpus → tweets_annotated_2.csv)
        ▼
04_discriminative_and_cooccurrence_analysis.py
        (loads the scored-tweets CSV as `df`; runs the
         ADHD-vs-ASD discriminative model and the symptom
         co-occurrence analysis across pre-filter thresholds)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for scripts 01–03.

## Data availability

**No data is included in this repository** — only code. Given that this
dataset combines self-disclosed ADHD/ASD status with inferred depression
symptom scores, even a tweet-ID-only release can act as a re-identification
pathway for a vulnerable population once IDs are hydrated back to accounts.
For that reason:

- Raw tweet text is never shared, consistent with X/Twitter's developer
  policy.
- The derived **tweet-ID + per-symptom-score CSV** (no raw text) is
  available **on request under a signed Data Use Agreement** — see
  [`DATA_USE_AGREEMENT.md`](DATA_USE_AGREEMENT.md) and
  [`data/README.md`](data/README.md) for details and how to request access.
- The source Twitter dataset and ReDSM5 corpus must be obtained
  independently under their own terms (see paper references 16, 17) to
  reproduce Stages 1–3 from scratch.

## Usage

### 1. Zero-shot depression-relevance pre-filter
```bash
python src/01_zero_shot_nli_prefilter.py
```
Edit `CSV_PATH` / `TEXT_COL` at the top of the script to point at your tweet CSV.

### 2. Grid search (Stage 2 classifier + per-label thresholds)
```bash
python src/02_mentalroberta_gridsearch_multilabel.py
```
Requires `redsm5_M5.csv` in the working directory. Edit `PARAM_GRID` to
change the search space; results are ranked by test macro-F1 in
`grid_search_outputs_multilabel/grid_search_results.csv`.

### 3. Final model + full-corpus inference
```bash
python src/03_final_train_and_tweet_inference.py
```
Set `TWEET_CSV` at the top of the script to your Stage-1-scored tweet CSV.
Trains on all labelled ReDSM5 data with the best hyperparameters from step 2,
then scores every tweet and writes per-symptom sigmoid scores only.

### 4. Group-level analysis
```python
import pandas as pd
df = pd.read_csv("tweets_annotated_2.csv")
exec(open("src/04_discriminative_and_cooccurrence_analysis.py").read())
```
Runs the L1-penalised logistic regression (ADHD vs ASD) with bootstrap
stability analysis, and the symptom co-occurrence (Pearson correlation)
analysis, both swept across five pre-filter thresholds (0.45–0.65).

## Citation

If you use this code, please cite the paper (see the manuscript for full
citation details once published).

This pipeline is trained on and derived from the following datasets —
please also cite these if you use them:

> Villa-Pérez ME, Trejo LA, Moin MB, Stroulia E. Extracting mental health
> indicators from English and Spanish social media: a machine learning
> approach. *IEEE Access*. 2023;11:128135–128152.

> Bao E, Pérez A, Parapar J. ReDSM5: a Reddit dataset for DSM-5 depression
> detection. In: *Proceedings of the 34th ACM International Conference on
> Information and Knowledge Management (CIKM '25)*. 2025.

## License

MIT (see `LICENSE`) — update if a different license is preferred.
