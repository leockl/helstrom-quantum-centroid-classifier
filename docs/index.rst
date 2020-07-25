.. Helstrom Quantum Centroid classifier documentation master file, created by
   sphinx-quickstart on Mon Jan 13 11:11:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Helstrom Quantum Centroid Classifier's Documentation!
====================================================================

The Helstrom Quantum Centroid (HQC) classifier is a quantum-inspired supervised classification approach for data with binary classes (ie. data with 2 classes only). By quantum-inspired, we mean a classification process which employs and exploits Quantum Theory.

It is inspired by the *quantum Helstrom observable* which acts on the distinguishability between quantum patterns rather than classical patterns of a dataset.

The HQC classifier is based on research undertaken by Giuseppe Sergioli, Roberto Giuntini and Hector Freytes, in their paper:

    Sergioli G, Giuntini R, Freytes H (2019) A new quantum approach to binary classification. PLoS ONE 14(5): e0216224.
    https://doi.org/10.1371/journal.pone.0216224

This Python package includes the option to vary four hyperparameters which are used to optimize the performance of the HQC classifier:

* There is an option to rescale the dataset.


* There are two possible options to choose how the classical dataset is encoded into quantum densities: *inverse of the standard stereographic projection* or *amplitude* encoding method.


* There is an option to choose the number of copies to take for the quantum densities.


* There are two possible options to assign class weights to the quantum Helstrom observable terms: *equiprobable* and *weighted*.



These hyperparameters are used in combination together to hypertune and optimize the accuracy of the HQC classifier to a given dataset.

The package also includes an option for parallel computing using Joblib and an option to split datasets into subsets or batches for optimal speed performance. Parallelization is performed over the two classes and subsets or batches.

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   user_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: General Example

   auto_examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
