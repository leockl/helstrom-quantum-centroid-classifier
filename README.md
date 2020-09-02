[![Build Status](https://travis-ci.com/leockl/helstrom-quantum-centroid-classifier.svg?branch=master)](https://travis-ci.com/leockl/helstrom-quantum-centroid-classifier)
[![Build status](https://ci.appveyor.com/api/projects/status/7lmgxf21o6atqs25?svg=true)](https://ci.appveyor.com/project/leockl/helstrom-quantum-centroid-classifier)
[![CircleCI](https://circleci.com/gh/leockl/helstrom-quantum-centroid-classifier.svg?style=svg)](https://circleci.com/gh/leockl/helstrom-quantum-centroid-classifier)
[![Documentation Status](https://readthedocs.org/projects/helstrom-quantum-centroid-classifier/badge/?version=latest)](https://helstrom-quantum-centroid-classifier.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/leockl/helstrom-quantum-centroid-classifier/branch/master/graph/badge.svg)](https://codecov.io/gh/leockl/helstrom-quantum-centroid-classifier)
[![PyPi Status](https://img.shields.io/pypi/v/HQC.svg?color=brightgreen)](https://pypi.org/project/HQC/)
[![Downloads](https://pepy.tech/badge/hqc)](https://pepy.tech/project/hqc)

# Helstrom Quantum Centroid Classifier
The Helstrom Quantum Centroid (HQC) classifier is a quantum-inspired supervised classification approach for data with binary classes (ie. data with 2 classes only). By quantum-inspired, we mean a classification process which employs and exploits Quantum Theory.

It is inspired by the *quantum Helstrom observable* which acts on the distinguishability between quantum patterns rather than classical patterns of a dataset.

The HQC classifier is based on research undertaken by Giuseppe Sergioli, Roberto Giuntini and Hector Freytes, in their paper:

    Sergioli G, Giuntini R, Freytes H (2019) A new quantum approach to binary classification. PLoS ONE 14(5): e0216224.
    https://doi.org/10.1371/journal.pone.0216224

This Python package includes the option to vary four hyperparameters which are used to optimize the performance of the HQC classifier:
* There is an option to rescale the dataset.
* There are two options to choose how the classical dataset is encoded into quantum densities: *inverse of the standard stereographic projection* or *amplitude* encoding method.
* There is an option to choose the number of copies to take for the quantum densities.
* There are two options to assign class weights to the quantum Helstrom observable terms: *equiprobable* or *weighted*.

These hyperparameters are used in combination together to hypertune and optimize the accuracy of the HQC classifier to a given dataset.

The package also includes an option for parallel computing using Joblib and an option to split datasets into subsets or batches for optimal speed performance. Parallelization is performed over the two classes and subset splits or batches.

It is shown in [the paper by Sergioli G, Giuntini R and Freytes H](https://doi.org/10.1371/journal.pone.0216224) that the HQC classifier, on average, **_outperforms_** a variety of commonly used classifiers over 14 real-world and artificial datasets, in particular when the hyperparameter, *number of copies*, is increased. A summary of the performances of the different classifiers examined are shown in the table below:

| Rank | Classifier                    | Average Success Rate (%) |
|:----:| ----------------------------- |:------------------------:|
| 1    | HelstromQuantumCentroid4      | 72.80                    |
| 2    | HelstromQuantumCentroid3      | 65.13                    |
| 3    | GaussianNB                    | 58.00                    |
| 4    | HelstromQuantumCentroid2      | 57.07                    |
| 5    | HelstromQuantumCentroid1      | 56.60                    |
| 5    | QuadraticDiscriminantAnalysis | 56.60                    |
| 6    | GradientBoostingClassifier    | 52.73                    |
| 7    | ExtraTreesClassifier          | 51.93                    |
| 8    | KNeighborsClassifier          | 51.47                    |
| 9    | NearestCentroid               | 49.13                    |
| 10   | RandomForestClassifier        | 45.87                    |
| 11   | QuantumNearestMeanCentroid    | 43.93                    |
| 12   | AdaBoostClassifier            | 42.93                    |
| 13   | LinearDiscriminantAnalysis    | 42.00                    |
| 14   | LogisticRegression            | 36.40                    |
| 15   | BernoulliNB                   | 17.40                    |

* Average success rate is the average number of datasets where the specified classifier outperforms the other classifiers over 14 real-world and artificial datasets. HelstromQuantumCentroidn is the HQC classifier corresponding to the n *number of copies* taken for the quantum densities.

The HQC classifier is a **_true probabilistic_** classifier, ie. a classifier which gives the class membership probability estimates for each sample without having to use any scaling methods. This means that the probability estimates (from `predict_proba`) are consistent with the scores (from `predict`).

## Source Code
The Python package's source code for the HQC classifier is available here: 
https://github.com/leockl/helstrom-quantum-centroid-classifier/blob/master/hqc/hqc.py

## Documentation
The documentation, including how to install the Python package, how to use the Python package and how the HQC classifier algorithm works, are available here: 
https://helstrom-quantum-centroid-classifier.readthedocs.io/en/latest/

## License
This Python package is licensed under the BSD 3-Clause License, available here: 
https://github.com/leockl/helstrom-quantum-centroid-classifier/blob/master/LICENSE

## References
Sergioli G, Giuntini R, Freytes H (2019) A new quantum approach to binary classification. PLoS ONE 14(5): e0216224.
https://doi.org/10.1371/journal.pone.0216224
