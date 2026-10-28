import pytest

from src.synthetic import make_synthetic_telco


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_telco(n=1500, seed=1)
