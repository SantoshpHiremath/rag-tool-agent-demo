# FordA Dataset — Reference Notes

## What FordA is

The FordA dataset is a univariate time-series classification dataset
from the UCR Time Series Archive, containing engine noise measurements
used to detect the presence or absence of a specific automotive
subsystem fault.

## Origin

FordA was contributed by Ford Motor Company as part of a 2008
classification competition and has since become a standard benchmark
for time-series classification research.

## Structure

Each FordA sample is a fixed-length sequence of 500 sensor readings,
labeled as either a normal engine measurement or one exhibiting the
target fault symptom. The training set contains 3601 instances and the
test set contains 1320 instances.

## Common preprocessing steps

Typical preprocessing for FordA before model training includes
z-normalizing each series (zero mean, unit variance) since the raw
sensor amplitude scale is not itself diagnostic, and confirming there
is no missing-value padding at the sequence boundaries, since UCR
archive exports occasionally include trailing NaNs for series shorter
than the archive's fixed length.

## Typical use cases

FordA is commonly used to benchmark time-series classifiers -- from
classical distance-based methods (e.g. Dynamic Time Warping with
k-nearest-neighbors) to deep learning approaches (1D CNNs, ResNets,
InceptionTime) -- because it is a real, moderately-sized, binary
classification problem with a well-established train/test split and
published baseline accuracies to compare against.
