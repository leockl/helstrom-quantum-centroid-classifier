import pytest

from sklearn.utils.estimator_checks import check_estimator

from hqc import hqc


@pytest.mark.parametrize(
    "Estimator", [hqc]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
