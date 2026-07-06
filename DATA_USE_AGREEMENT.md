# Data Use Agreement

**Dataset:** Derived tweet-ID + per-symptom-score dataset accompanying
*"Population-Level Profiling of DSM-5 Depressive Symptoms Among
Self-Reported ADHD and ASD Users on Twitter: An Exploratory Study Using
Advanced NLP and Statistical Analysis."*

**Data Provider:** [Your Name], [Your Institution] (corresponding author of
the associated paper)

This Agreement governs access to the derived dataset described below,
between the Data Provider and the individual or institution requesting
access ("the Recipient"). By requesting, receiving, or using the dataset,
the Recipient agrees to the terms in this Agreement.

## 1. Description of the data

The dataset contains, per tweet:

| Field | Description |
|---|---|
| `tweet_id` | Original platform tweet identifier |
| `user_id` | Pseudonymous, study-internal grouping key (does not resolve directly to a platform account) |
| `disorder` | Self-reported group label (ADHD / ASD) |
| `predicted_prob` | Stage 1 zero-shot depression-relevance score |
| `score_<SYMPTOM>` | Nine Stage 2 per-tweet DSM-5 symptom sigmoid scores |

**No raw tweet text, usernames, or other direct identifiers are included.**
The data is derived from a Twitter mental-health dataset obtained via IEEE
DataPort (Villa-Pérez et al., 2023 — see Section 5) and from model
inference performed by the Data Provider; it is not redistributed from the
original source in its raw form.

## 2. Permitted use

The dataset may be used **solely for academic, non-commercial research
purposes**, including verification or extension of the results reported in
the associated paper. Any other use (including commercial use) requires
separate written permission from the Data Provider.

## 3. Recipient's obligations

By accessing this dataset, the Recipient agrees to:

1. **No re-identification.** Not attempt to identify, hydrate, contact, or
   otherwise single out any individual user represented in the dataset,
   and not combine this dataset with any other data source for the purpose
   of re-identifying individuals.
2. **No redistribution.** Not share, publish, sub-license, or transfer the
   dataset, in whole or in part, to any third party — including via public
   code/data repositories, cloud storage links, or academic data-sharing
   platforms — without the Data Provider's prior written consent.
3. **Independent compliance.** Independently ensure that their own
   collection, storage, and use of any data derived from tweet IDs (should
   the Recipient choose to hydrate them) complies with the current
   developer policy of the source platform (X/Twitter) and with the access
   terms of the original IEEE DataPort dataset.
4. **Secure storage & deletion.** Store the dataset on access-controlled,
   institutionally-approved infrastructure, and delete all copies within
   **24 months** of receipt, or upon completion of the stated research use,
   whichever is sooner — unless a longer retention period is agreed with
   the Data Provider in writing in advance.
5. **Citation.** Cite the associated paper (and, where the underlying
   ReDSM5 annotations are used, Bao, Pérez & Parapar, 2025 — see Section 5)
   in any publication, presentation, dataset, model, or other work product
   that makes use of this dataset.
6. **Institutional approval.** Confirm that their institution's IRB/ethics
   review board has reviewed or exempted this secondary-data use, and
   provide evidence of this determination to the Data Provider upon
   request.
7. **Notification of incidents.** Notify the Data Provider promptly if the
   Recipient becomes aware of any unauthorized access to, or disclosure of,
   the dataset.

## 4. Disclaimer and liability

The dataset is provided "as is," without warranty of any kind. The Data
Provider makes no representation regarding the continued existence,
accuracy, or availability of the underlying tweets, which may be deleted,
edited, or made private by users or the platform at any time. The Data
Provider is not liable for any use the Recipient makes of the dataset
beyond the terms of this Agreement.

## 5. Source data citations

> Villa-Pérez ME, Trejo LA, Moin MB, Stroulia E. Extracting mental health
> indicators from English and Spanish social media: a machine learning
> approach. *IEEE Access*. 2023;11:128135–128152.

> Bao E, Pérez A, Parapar J. ReDSM5: a Reddit dataset for DSM-5 depression
> detection. In: *Proceedings of the 34th ACM International Conference on
> Information and Knowledge Management (CIKM '25)*. 2025.

## 6. Term

This Agreement takes effect upon the Recipient's receipt of the dataset and
remains in force until the retention period in Section 3.4 ends, or until
terminated earlier by either party in writing.

---

### Recipient details

Name: _______________________
Institution: _______________________
Email: _______________________
Intended use (brief description): _______________________
IRB/ethics approval reference (if applicable): _______________________

Signature: _______________________          Date: _______________

### Data Provider

Signature: _______________________          Date: _______________

