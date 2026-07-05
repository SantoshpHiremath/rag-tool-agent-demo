# FordA Dataset — Reference Notes

## Overview
The FordA dataset is a widely used benchmark dataset in time-series classification research,
originally released as part of the UCR Time Series Classification Archive. It was created in
collaboration with Ford Motor Company for a competition on automotive engine diagnostics.

## Purpose
The dataset's task is to detect the presence (or absence) of a specific automotive subsystem
fault based on symptoms encoded in an engine noise/vibration signal. Each sample is a single
univariate time series representing a measurement captured over a fixed time window from a
sensor in an automotive subsystem.

## Structure
- **Classes**: Binary classification — class +1 indicates a fault symptom is present in the
  measurement, class -1 indicates no fault symptom.
- **Series length**: Each time series instance has 500 data points.
- **Training set**: 3,601 instances.
- **Test set**: 1,320 instances.
- **Domain**: Automotive engine diagnostics, industrial sensor monitoring.

## Common Preprocessing Steps
1. **Noise reduction** — automotive sensor signals are prone to high-frequency noise from
   engine vibration; smoothing filters (e.g. moving average, Savitzky-Golay) are commonly
   applied before feature extraction.
2. **Normalization** — z-score normalization per series is standard practice, since raw
   amplitude varies across sensors and recording sessions.
3. **Lag-based feature engineering** — deriving lagged versions of the signal (t-1, t-5, t-10)
   helps supervised models capture temporal dependencies without needing a full sequence model.
4. **Synthetic target generation** — for derivative tasks (e.g. predicting RPM/speed instead of
   raw fault classification), synthetic targets are generated from signal characteristics such
   as dominant frequency or periodicity, since the original dataset does not include RPM labels
   directly.

## Typical Use Cases
- Binary fault classification (the original UCR task).
- Benchmarking time-series classifiers (e.g. shapelets, ROCKET, InceptionTime, 1D-CNNs).
- Derivative supervised learning tasks built on top of the raw signal, such as speed/RPM
  estimation, when combined with synthetic target engineering.

## Known Challenges
- High intra-class variability due to different engines and operating conditions.
- Sensitivity of downstream model accuracy to preprocessing choices, particularly noise
  reduction and normalization.
- Class imbalance is mild but should be checked per split before training.
