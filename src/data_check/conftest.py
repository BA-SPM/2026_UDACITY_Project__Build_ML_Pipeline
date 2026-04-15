import pytest
import pandas as pd
import wandb
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    # OLD: data_path = run.use_artifact(request.config.option.csv).file()
    # Windows note: .file() can create a local path containing :latest, which is
    # not a valid Windows path segment.
    input_artifact = run.use_artifact(request.config.option.csv)
    artifact_dir = Path(input_artifact.download())
    csv_files = sorted(artifact_dir.glob("*.csv"))

    if not csv_files:
        pytest.fail("You must provide the --csv option on the command line")

    data_path = csv_files[0]
    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    # OLD: data_path = run.use_artifact(request.config.option.ref).file()
    # Windows note: .file() can create a local path containing :reference, which
    # is not a valid Windows path segment.
    ref_artifact = run.use_artifact(request.config.option.ref)
    artifact_dir = Path(ref_artifact.download())
    csv_files = sorted(artifact_dir.glob("*.csv"))

    if not csv_files:
        pytest.fail("You must provide the --ref option on the command line")

    data_path = csv_files[0]
    df = pd.read_csv(data_path)

    return df


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
