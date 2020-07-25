import pytest

from sklearn.utils.estimator_checks import check_estimator

from hqc import HQC


@pytest.mark.parametrize(
    "Estimator", [hqc]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
