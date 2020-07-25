User Guide
==========

This user guide will show you a step-by-step guide on how to install the Helstrom Quantum Centroid (HQC) classifier's API, how to use the HQC classifier's API and how the algorithm behind the HQC classifier works. 

How to install the API
----------------------

To install the HQC classifier's Python package, you will first need to install a Python-based package manager called Pip. See instructions `in this link <https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation>`_ (under 'Pip install') on how to install Pip.

Once you have Pip installed, open the command prompt (or cmd) window on your computer. Change directories to the folder that contains the Python file ``get-pip.py`` that you have just downloaded. Then run the following command in the command prompt:

.. code:: bash

    

    pip install hqc

You have now installed the HQC classifier's Python package in your computer and you are now able to use it in Python Interpreter.

You can also install the HQC classifier's Python package in Anaconda. Pip is already pre-installed in Anaconda so you do not need to install Pip if you are using Anaconda. To install the HQC classifier's Python package in Anaconda, open the Anaconda Prompt window and run the following command:

.. code:: bash

    

    pip install hqc

You have now installed the HQC classifier's Python package in Anaconda and you are now able to use it in Anaconda.

Tip: If there is a newer version of the HQC classifier's Python package released, you can run the following command in the command prompt (cmd) or Anaconda Prompt window to upgrade to a newer version:

.. code:: bash

    

    pip install hqc --upgrade

How to use the API
------------------

First, import the HQC classifier's Python package into Python Interpreter or Anaconda:

.. code:: python

    import hqc

Then specify values for the four hyperparameters for the HQC classifier:

1. the rescaling factor, *rescale*.

2. the encoding method, *encoding*.

3. the number of copies to take for each quantum density, *n_copies*. 

4. the class weights assigned to the quantum Helstrom observable terms, *class_wgt*.

If you wish to perform parallel computing, you can also specify values for the following two parameters:

1. the number of CPU cores used when parallelizing, *n_jobs*.

2. the number of subset or batches splits performed on the datasets, *n_splits*.

Say, *rescale* = 1.5, *encoding* = stereo, *n_copies* = 2, *class_wgt* = weighted, *n_jobs* = 4 and *n_splits* = 2, the HQC classifier would look like:

.. code:: python

    hqc.HQC(rescale=1.5, encoding='stereo', n_copies=2, class_wgt='weighted', n_jobs=4, n_splits=2)

If either *rescale*, *n_copies*, *n_jobs* and/or *n_splits* are not specified, they would default to 1. If *encoding* is not specified, it would default to amplit. If *class_wgt* is not specified, it would default to equi. The HQC classifier would look like:

.. code:: python

    hqc.HQC()

where *rescale* = 1, *encoding* = 'amplit', *n_copies* = 1, *class_wgt* = 'equi', *n_jobs* = 1 and *n_splits* = 1.

From here on, we will be using *rescale* = 1.5, *encoding* = stereo, *n_copies* = 2, *class_wgt* = weighted, *n_jobs* = 4 and *n_splits* = 2 as an example. To get your HQC classification model, fit the features matrix X and binary target vector y, as below:

.. code:: python

    model = hqc.HQC(rescale=1.5, encoding='stereo', n_copies=2, class_wgt='weighted', n_jobs=4, n_splits=2).fit(X, y)

Tip: If the feature matrix X contains non-numerical categorical features, these features could first be encoded into numerical 0s and 1s using the one-hot encoding method. The binary target vector y can be a numerical feature of any numbers as long as it has only two classes.

The fitted attributes of your model can be obtained by calling the following methods:

=======================   ============================================================================================================
Method                    Fitted Attribute
=======================   ============================================================================================================
model.classes_            Gives the sorted binary classes.
model.centroids_          Gives the Quantum Centroids for the two classes, with index 0 and 1 respectively.
model.hels_obs_           Gives the Quantum Helstrom observable.
model.proj_sums_          Gives the sum of the projectors of the Quantum Helstrom observable's eigenvectors, which has corresponding positive and negative eigenvalues respectively.
model.hels_bound_         Gives the Helstrom bound.
=======================   ============================================================================================================

For prediction, you can obtain the trace matrix where column index 0 corresponds to the trace values for class 0 and column index 1 corresponds to the trace values for class 1, by using:

.. code:: python

    model.predict_proba(X)

You can then obtain the class predictions by using:

.. code:: python

    model.predict(X)

You can obtain the accuracy score by using:

.. code:: python

    model.score(X, y)

You can use scikit-learn's GridSearchCV tool to do an exhaustive search to find the optimal values for the hyperparameters *rescale*, *encoding*, *n_copies* and *class_wgt*. For eg.:

.. code:: python

    from sklearn.model_selection import GridSearchCV
    import pandas as pd

    param_grid = {'rescale':[0.5, 1, 1.5], 'encoding':['amplit', 'stereo'], 'n_copies':[1, 2], 'class_wgt':['equi', 'weighted']}
    models = GridSearchCV(hqc.HQC(), param_grid).fit(X, y)

    # To ouput a dataframe table of all the models specified in param_grid
    pd.DataFrame(models.cv_results_)

More information about scikit-learn's GridSearchCV tool can be found `here <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.        

How does the HQC classifier algorithm works
-------------------------------------------

Below is a general step-by-step guide to the algorithm behind the HQC classifier. The source code can be found in this `link <https://github.com/leockl/helstrom-quantum-centroid-classifier/blob/master/hqc/hqc.py>`_.

1. First the algorithm perform checks on the features matrix X and binary target vector y, such as checking if X and y have the same number of rows (samples/observations) and y is of a categorical type variable.

2. Then it encodes y into binary classes 0 and 1.

3. Because the following calculations from here on would involve decimal places, X is converted to float to allow for floating point calculations.

4. X is then multiplied by the rescaling factor, *rescale* (chosen by the user).

5. The algorithm then calculates the sum of squares for each row (sample/observation) in X.

6. Next the algorithm determines the number of rows and columns in X.

7. X' is then calculated according to the encoding method chosen by the user.

8. After this, the algorithm calculates the terms in the Quantum Centroids and Quantum Helstrom observable for each of the two classes.

9. For X' with binary class 0, the algorithm first determines and then calculates the number of rows or samples in X' belonging to this class. The algorithm then splits this dataset belonging to this class into subsets or batches according to the number of splits specified by the user. 

10. For each subset or batch, the algorithm then calculates the terms in the Quantum Centroids and quantum Helstrom observable by combining these steps: first creating a counter to identify each subset or batch and then calculating the number of rows or samples in this subset or batch. The algorithm then determines the number of rows, which is equivalent to the number of columns of the density_sum, centroids and hels_obs_terms arrays (since they are symmetric matrices) and initializes these arrays. Next, the algorithm calculates the quantum densities for each row (sample/observation), then calculate the *n_copies* or the n-fold Kronecker product (chosen by the user) for each quantum density and then summing the n-fold quantum densities and deviding by the number of rows (samples/observations) to get the quantum centroid for each subset or batch, for X' with binary class 0. The algorithm also calculates the terms in the quantum Helstrom observable according to the *class_wgt* option chosen by the user, for each subset or batch, for X' with binary class 0. Parallelization is performed over each of these subsets or batches, and the quantum centroid and quantum Helstrom observable terms for binary class 0 are obtained by summing the quantum centroid and quantum Helstrom observable terms for all the subsets or batches for binary class 0.

11. Steps 9 and 10 are repeated for X' in the group with binary class 1, using parallelization.

12. Next, the algorithm calculates the *quantum Helstrom observable* matrix.

13. This is followed by determining the eigenvalues and eigenvectors of the *quantum Helstrom observable* matrix as well as determining the number of eigenvalues.

14. To determine the eigenvectors corresponding to positive or negative eigenvalues, the algorithm first creates an array of 0s and 1s to indicate positive or negative eigenvalues and then tranposes the matrix containing the eigenvectors to row-wise in order to determine which row in the matrix containing the eigenvectors belonging to positive or negative eigenvalues.

15. The algorithm then splits the matrix containing the eigenvectors corresponding to positive eigenvalues into subsets or batches according to the number of splits specified by the user. 

16. Next the algorithm sums all the projectors for each subset or batch for eigenvectors corresponding to positive eigenvalues. The projector of an eigenvector is defined as the dot product between the unit eigenvector and its transpose, ie. ``np.dot(v, np.transpose(v))`` where v is a column vector of the unit eigenvector. The projector of an eigenvector is a matrix. Parallelization is performed over each of these subsets or batches, and the sum of all the projectors for eigenvectors corresponding to positive eigenvalues is obtained by summing the sum of the projectors for all the subsets or batches for eigenvectors corresponding to positive eigenvalues.

17. Steps 15 and 16 are repeated for eigenvectors corresponding to negative eigenvalues, using parallelization.

18. Now, the algorithm calculates the Helstrom bound.

19. Moving into prediction, the algorithm first perform checks such as to see if a model have already been fitted and the matrix X we are predicting on has the same number of columns as the features matrix X. 

20. The algorithm then repeats Steps 3, 4, 5, 6 and 7 (with no seperation into the two groups, binary class 0 and binary class 1).

21. Next, the algorithm calculates the trace values corresponding to binary class 0 by first splitting the datasets into subsets or batches according to the number of splits specified by the user. 

22. For each subset or batch corresponding to binary class 0, the algorithm calculates the trace values by combining these steps: first creating a counter to identify each subset or batch and then calculating the number of rows or samples in this subset or batch. Then the algorithm calculates the quantum densities for each row (sample/observation) in the subset or batch that we want to predict, followed by calculating the *n_copies* or the n-fold Kronecker product (chosen by the user) for each quantum density. The trace values of the dot product between the n-fold quantum densities and the sum of projectors with corresponding positive eigenvalues is then calculated for each subset or batch. The trace values corresponding to binary class 0 is then obtained by concatenating the trace values for all of the subsets or batches corresponding to binary class 0.

23. Steps 21 and 22 are repeated to calculate the trace values corresponding to binary class 1, using parallelization. The trace values corresponding to binary class 0 and class 1 are placed in a trace matrix.

#. Finally, the algorithm determines the predicted binary class 0 or 1 by comparing the trace values in the trace matrix. If the trace values in the first column of the trace matrix is higher than (or the same as) the trace values in the second column, the predicted binary class is 0, otherwise the predicted binary class is 1.
