# Classification

## Candidate Methods

- Common Spatial Patterns (CSP)
- Filter Bank Common Spatial Patterns (FBCSP)
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)
- Naive Bayes / Parzen window classifiers
- Logistic regression
- Riemannian covariance / tangent-space classifiers

## Current Working Choice

The current default classical decoder is:

```text
FBCSP + shrinkage LDA
```

Reason:

- It is literature-compatible for motor-imagery EEG.
- It performed best on the current BCI Competition IV Dataset 2b benchmark on
  average.
- It is explainable enough for the dissertation methods chapter.
- It is lightweight enough for online use.

Riemannian tangent-space models remain implemented as comparison candidates:

```text
covariance -> tangent space -> LDA / logistic regression
```

They are kept because they perform competitively and can be better for some
subjects.

Additional implemented comparison models:

```text
CSP + SVM
CSP + logistic regression
CSP + Parzen window classifier
CSP + Naive Bayes Parzen Window classifier
FBCSP + SVM
FBCSP + logistic regression
FBCSP + Parzen window classifier
FBCSP + Naive Bayes Parzen Window classifier
FBCSP + MIBIF feature selection + Naive Bayes Parzen Window classifier
```

SVM is included because CSP/FBCSP with SVM is common in motor-imagery BCI
literature. It is useful as a benchmark candidate, but the final online model
should still be chosen using user-specific REST-vs-ACTIVE calibration data and
false-positive behavior, not accuracy alone.

`FBCSP + Naive Bayes Parzen Window` and `FBCSP + MIBIF + NBPW` are included
because the top BCI Competition IV Dataset 2b submission used FBCSP with a
Naive Bayes Parzen Window classifier. The local implementation uses independent
one-dimensional Gaussian kernel densities per selected feature and class.

## Candidate Pipeline

1. EEG acquisition
2. filtering, usually motor-imagery bands such as 8-30 Hz or a filter bank
3. artifact handling / rejected-trial exclusion
4. feature extraction
5. REST-vs-ACTIVE detection
6. state-machine gated command generation

## Current Decision

The first robot-control version should use:

```text
REST vs ACTIVE/MI
```

Mapping:

```text
REST -> no command / hold
ACTIVE/MI -> confirm, only in valid FSM states
```

Left-vs-right remains useful for public-data benchmarking and later upgrade
paths, but it is not the first online robot command scheme. The reason is
false-positive risk: the robot needs a reliable no-command state before adding
multiple MI commands.

## Public Dataset Benchmark Notes

BCI Competition IV Dataset 2b / Graz B was used to validate the decoding
pipeline:

- left/right benchmark: cue-based `769` vs `770`;
- pseudo REST/ACTIVE benchmark: left/right MI pooled as ACTIVE, safe
  baseline/idling windows used as REST;
- rejected/artifact trials marked with `1023` are excluded;
- EOG channels are not used for classification.

Interpretation:

- `FBCSP+LDA` is the current default candidate.
- Riemannian models are retained as subject-dependent alternatives.
- Pseudo REST/ACTIVE performance is a warning signal only; final model
  selection must use real Unicorn REST/ACTIVE calibration data.

## Online Decision Rule

Classifier output is not directly a robot command:

```text
EEG window
-> P(ACTIVE)
-> confidence threshold
-> majority vote
-> cooldown
-> FSM gate
-> MI_CONFIRM or NO_COMMAND
```

This distinction matters because the raw classifier may output ACTIVE during
idle windows. The robot should only receive commands after state-machine
gating.

## Future / Risk Note

EEG non-stationarity may require adaptive approaches if performance drifts
across blocks or sessions. Keep adaptation as a contingency plan rather than
the first implementation.

## Notes

A simpler REST-vs-ACTIVE paradigm is expected to be more robust for early
real-time teleoperation tests than multi-command left/right MI.
