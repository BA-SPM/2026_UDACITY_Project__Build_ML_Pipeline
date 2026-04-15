import pytest
import pandas as pd
from pathlib import Path

# Path to the local fixture files (relative to this conftest)
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_DEFAULT_CSV = str(_FIXTURES_DIR / "clean_sample.csv")
_DEFAULT_REF = str(_FIXTURES_DIR / "clean_reference.csv")


def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default=_DEFAULT_CSV)
    parser.addoption("--ref", action="store", default=_DEFAULT_REF)
    parser.addoption("--kl_threshold", action="store", default="0.2")
    parser.addoption("--min_price", action="store", default="10")
    parser.addoption("--max_price", action="store", default="350")


@pytest.fixture(scope='session')
def data(request):
    """Load sample CSV from local fixtures. No W&B required."""
    csv_path = Path(request.config.option.csv)

    if not csv_path.exists():
        pytest.fail(f"CSV fixture not found: {csv_path}")

    return pd.read_csv(csv_path)


@pytest.fixture(scope='session')
def ref_data(request):
    """Load reference CSV from local fixtures. No W&B required."""
    ref_path = Path(request.config.option.ref)

    if not ref_path.exists():
        pytest.fail(f"Reference CSV fixture not found: {ref_path}")

    return pd.read_csv(ref_path)


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
