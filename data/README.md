# Data

**No data is hosted in this repository.** Consistent with the paper's
Ethics Statement and the sensitivity of this dataset (self-disclosed ADHD/
ASD status combined with inferred depression-symptom scores), tweet IDs and
per-symptom scores are **not published on GitHub**. Even ID-only sharing can
function as a re-identification pathway for a vulnerable population once
combined with disorder labels and symptom inferences — see the note in the
main README's "Data availability" section for the reasoning.

## Requesting the derived (tweet_id + score) dataset

The tweet-ID + per-symptom-score CSV (no raw tweet text) described in
`04_discriminative_and_cooccurrence_analysis.py` is available **on request,
under a signed Data Use Agreement (DUA)**. See
[`DATA_USE_AGREEMENT.md`](../DATA_USE_AGREEMENT.md) for the full agreement,
and contact the corresponding author (see the paper) to initiate a request.

Typical conditions requesters agree to:
- No attempt to re-identify individual users.
- No further redistribution of the data.
- Independent compliance with the source platform's (X/Twitter) current
  developer policy and the original IEEE DataPort dataset's access terms.
- Deletion of the data once the stated research use concludes.
- Citation of the paper in any resulting publication.

## Source datasets

- **Twitter mental-health dataset**: publicly available via IEEE DataPort.
  Not redistributed here — obtain directly from the source under its own
  terms.
  > Villa-Pérez ME, Trejo LA, Moin MB, Stroulia E. Extracting mental health
  > indicators from English and Spanish social media: a machine learning
  > approach. *IEEE Access*. 2023;11:128135–128152.

- **ReDSM5**: expert-annotated Reddit corpus used to train the Stage 2
  classifier (`redsm5_M5.csv`, required by scripts 02 and 03). Obtain from
  the original authors.
  > Bao E, Pérez A, Parapar J. ReDSM5: a Reddit dataset for DSM-5 depression
  > detection. In: *Proceedings of the 34th ACM International Conference on
  > Information and Knowledge Management (CIKM '25)*. 2025.

## Expected file for scripts 02/03

`redsm5_M5.csv` — ReDSM5 corpus formatted with columns:
`sentence_id, sentence_text, DSM5_symptom, status`

## Expected file for script 04

A scored-tweets CSV (output of script 03) with columns:
`tweet_id, user_id, disorder, predicted_prob, score_DEPRESSED_MOOD, ...`
(see the docstring at the top of `04_discriminative_and_cooccurrence_analysis.py`)
