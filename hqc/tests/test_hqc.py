import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer

from hqc import HQC


# Check classifier's hyperparameters, attributes and predict method have been created 
@pytest.fixture
def data():
    return load_breast_cancer(return_X_y=True)

def test_template_classifier(data):
    X, y = data
    clf = HQC()
    assert clf.rescale == 1
    assert clf.encoding == 'amplit'
    assert clf.n_copies == 1
    assert clf.class_wgt == 'equi'
    assert clf.n_jobs == None
    assert clf.n_splits == 1

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'centroids_')
    assert hasattr(clf, 'hels_obs_')
    assert hasattr(clf, 'proj_sums_')
    assert hasattr(clf, 'hels_bound_')
    
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    
    
# Load breast cancer dataset and randomly permute it
breast_cancer = load_breast_cancer()
rng = np.random.RandomState(1)
perm = rng.permutation(breast_cancer.target.size)
breast_cancer.data = breast_cancer.data[perm]
breast_cancer.target = breast_cancer.target[perm]

# Check consistency of different hyperparameters on breast cancer dataset
def test_breast_cancer():
    helstrom_bound = []
    true_result = [0.5531, 0.5531]
    for rescale in [0.5, 1]:
        clf = HQC(rescale=rescale)
        clf.fit(breast_cancer.data, breast_cancer.target)
        helstrom_bound.append(clf.hels_bound_)
    assert_array_equal(np.round(helstrom_bound, 4), true_result)
    
    helstrom_bound = []
    true_result = [0.5531, 0.5008]
    for encoding in ['amplit', 'stereo']:
        clf = HQC(encoding=encoding)
        clf.fit(breast_cancer.data, breast_cancer.target)
        helstrom_bound.append(clf.hels_bound_)
    assert_array_equal(np.round(helstrom_bound, 4), true_result)    

    helstrom_bound = []
    true_result = [0.5531, 0.5744]
    for n_copies in [1, 2]:
        clf = HQC(n_copies=n_copies)
        clf.fit(breast_cancer.data, breast_cancer.target)
        helstrom_bound.append(clf.hels_bound_)
    assert_array_equal(np.round(helstrom_bound, 4), true_result)  
    
    helstrom_bound = []
    true_result = [0.5531, 0.6369]
    for class_wgt in ['equi', 'weighted']:
        clf = HQC(class_wgt=class_wgt)
        clf.fit(breast_cancer.data, breast_cancer.target)
        helstrom_bound.append(clf.hels_bound_)
    assert_array_equal(np.round(helstrom_bound, 4), true_result)  
    
    
# Create toy dataset
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# Check consistency of classification on toy dataset
def test_classification_toy():
    clf = HQC()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
