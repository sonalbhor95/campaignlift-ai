# Model Card — CampaignLift AI

## Intended use

This project demonstrates causal uplift modeling for marketing targeting. It is intended for portfolio and learning purposes.

## Dataset

Criteo Uplift Prediction Dataset, which includes anonymized user features, treatment indicator, exposure indicator, visit label, and conversion label.

## Targets

The default target is `visit` because it is less sparse for local experimentation. `conversion` is also supported but requires a larger sample.

## Models

- Response model baseline
- S-Learner uplift model
- T-Learner uplift model

## Evaluation

- Response model ROC-AUC and average precision
- Qini-style uplift AUC
- Decile-level observed uplift
- Campaign targeting profit simulation

## Limitations

- Criteo features are anonymized, so business feature interpretation is limited.
- Uplift evaluation is sensitive to treatment/control balance and sparse outcomes.
- The simplified Qini implementation is suitable for portfolio demonstration but should be validated before production use.
- Dataset terms are non-commercial/share-alike; review original terms before commercial use.
