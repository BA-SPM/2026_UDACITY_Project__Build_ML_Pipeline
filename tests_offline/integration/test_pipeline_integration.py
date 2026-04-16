"""Offline integration tests for the local pipeline wiring.

These tests keep all external services mocked and validate that a small offline
pipeline flow behaves as expected.
"""

from argparse import Namespace
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

import main as pipeline_main
from src.basic_cleaning import run as basic_cleaning_run
from src.data_check import test_data as online_data_checks


class FakeInputArtifact:
    """Return a local directory when the cleaning step requests an artifact."""

    def __init__(self, directory: Path):
        self.directory = directory

    def download(self) -> str:
        """Return the mocked artifact directory path."""
        return str(self.directory)


class FakeOutputArtifact:
    """Capture artifact metadata without calling W&B services."""

    def __init__(self, name: str, type: str, description: str):
        self.name = name
        self.type = type
        self.description = description
        self.files: list[str] = []

    def add_file(self, file_path: str) -> None:
        """Store the file path that would be attached to the artifact."""
        self.files.append(file_path)


class FakeRun:
    """Minimal W&B run stub for offline integration tests."""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.config = FakeConfig()
        self.logged_artifact: FakeOutputArtifact | None = None

    def use_artifact(self, _artifact_name: str) -> FakeInputArtifact:
        """Return the local artifact directory for the cleaning step."""
        return FakeInputArtifact(self.input_dir)

    def log_artifact(self, artifact: FakeOutputArtifact) -> None:
        """Capture the artifact that would have been logged."""
        self.logged_artifact = artifact


class FakeConfig(dict):
    """Minimal config object that accepts argparse namespaces."""

    def update(self, values) -> None:  # type: ignore[override]
        """Store namespace values in dictionary form."""
        if isinstance(values, Namespace):
            super().update(vars(values))
            return

        super().update(values)


def _build_raw_dataframe() -> pd.DataFrame:
    """Create a realistic raw dataset large enough for the online data checks."""
    template_rows = [
        {
            "id": 1,
            "name": "Bronx apartment",
            "host_id": 101,
            "host_name": "Host A",
            "neighbourhood_group": "Bronx",
            "neighbourhood": "Fordham",
            "latitude": 40.86,
            "longitude": -73.89,
            "room_type": "Private room",
            "price": 55,
            "minimum_nights": 2,
            "number_of_reviews": 10,
            "last_review": "2019-01-01",
            "reviews_per_month": 1.2,
            "calculated_host_listings_count": 1,
            "availability_365": 180,
        },
        {
            "id": 2,
            "name": "Brooklyn loft",
            "host_id": 102,
            "host_name": "Host B",
            "neighbourhood_group": "Brooklyn",
            "neighbourhood": "Williamsburg",
            "latitude": 40.71,
            "longitude": -73.96,
            "room_type": "Entire home/apt",
            "price": 120,
            "minimum_nights": 3,
            "number_of_reviews": 25,
            "last_review": "2019-02-15",
            "reviews_per_month": 2.5,
            "calculated_host_listings_count": 2,
            "availability_365": 200,
        },
        {
            "id": 3,
            "name": "Manhattan studio",
            "host_id": 103,
            "host_name": "Host C",
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Harlem",
            "latitude": 40.81,
            "longitude": -73.95,
            "room_type": "Entire home/apt",
            "price": 180,
            "minimum_nights": 2,
            "number_of_reviews": 40,
            "last_review": "2019-03-22",
            "reviews_per_month": 3.8,
            "calculated_host_listings_count": 1,
            "availability_365": 150,
        },
        {
            "id": 4,
            "name": "Queens room",
            "host_id": 104,
            "host_name": "Host D",
            "neighbourhood_group": "Queens",
            "neighbourhood": "Astoria",
            "latitude": 40.76,
            "longitude": -73.92,
            "room_type": "Private room",
            "price": 75,
            "minimum_nights": 1,
            "number_of_reviews": 12,
            "last_review": "2019-04-18",
            "reviews_per_month": 1.4,
            "calculated_host_listings_count": 1,
            "availability_365": 220,
        },
        {
            "id": 5,
            "name": "Staten Island home",
            "host_id": 105,
            "host_name": "Host E",
            "neighbourhood_group": "Staten Island",
            "neighbourhood": "St. George",
            "latitude": 40.64,
            "longitude": -74.08,
            "room_type": "Entire home/apt",
            "price": 95,
            "minimum_nights": 2,
            "number_of_reviews": 8,
            "last_review": "2019-05-11",
            "reviews_per_month": 0.9,
            "calculated_host_listings_count": 1,
            "availability_365": 250,
        },
    ]

    repeated_rows = template_rows * 4000
    outlier_rows = [
        {**template_rows[0], "id": 900001, "price": 5},
        {**template_rows[1], "id": 900002, "price": 500},
    ]

    return pd.DataFrame(repeated_rows + outlier_rows)


def test_basic_cleaning_output_passes_online_data_checks(tmp_path, monkeypatch):
    """Run the cleaning step offline and validate the output with online assertions."""
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    _build_raw_dataframe().to_csv(raw_dir / "sample.csv", index=False)

    fake_run = FakeRun(raw_dir)
    monkeypatch.setattr(basic_cleaning_run.wandb, "init", lambda job_type: fake_run)
    monkeypatch.setattr(basic_cleaning_run.wandb, "Artifact", FakeOutputArtifact)
    monkeypatch.chdir(tmp_path)

    args = Namespace(
        input_artifact="sample.csv:latest",
        output_artifact="clean_sample.csv",
        output_type="clean_sample",
        output_description="Offline_pipeline_integration_test",
        min_price=10.0,
        max_price=350.0,
    )

    basic_cleaning_run.go(args)

    cleaned_df = pd.read_csv(tmp_path / "clean_sample.csv")

    online_data_checks.test_column_names(cleaned_df)
    online_data_checks.test_neighborhood_names(cleaned_df)
    online_data_checks.test_proper_boundaries(cleaned_df)
    online_data_checks.test_similar_neigh_distrib(cleaned_df, cleaned_df.copy(), 0.2)
    online_data_checks.test_row_count(cleaned_df)
    online_data_checks.test_price_range(cleaned_df, 10.0, 350.0)

    assert fake_run.logged_artifact is not None
    assert fake_run.logged_artifact.files == ["clean_sample.csv"]


def test_main_orchestrates_requested_pipeline_steps(monkeypatch):
    """Verify that the offline pipeline orchestration wires the expected steps."""
    recorded_calls: list[tuple[str, str, str, dict[str, object]]] = []

    def fake_mlflow_run(component_uri, entry_point, env_manager, parameters):
        recorded_calls.append((component_uri, entry_point, env_manager, parameters))
        return object()

    monkeypatch.setattr(pipeline_main.mlflow, "run", fake_mlflow_run)
    monkeypatch.setattr(
        pipeline_main.hydra.utils,
        "get_original_cwd",
        lambda: "C:/offline-pipeline",
    )

    config = OmegaConf.create(
        {
            "main": {
                "components_repository": "components",
                "project_name": "nyc_airbnb",
                "experiment_name": "offline_integration",
                "steps": "basic_cleaning,data_check",
            },
            "etl": {"sample": "sample1.csv", "min_price": 10, "max_price": 350},
            "data_check": {"kl_threshold": 0.2},
            "modeling": {
                "test_size": 0.2,
                "val_size": 0.2,
                "random_seed": 42,
                "stratify_by": "neighbourhood_group",
                "max_tfidf_features": 30,
                "random_forest": {},
            },
        }
    )

    pipeline_main.go.__wrapped__(config)

    normalized_calls = [
        (Path(component_uri).as_posix(), entry_point, env_manager, parameters)
        for component_uri, entry_point, env_manager, parameters in recorded_calls
    ]

    assert normalized_calls == [
        (
            "C:/offline-pipeline/src/basic_cleaning",
            "main",
            "local",
            {
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data_with_outliers_and_null_values_removed",
                "min_price": 10,
                "max_price": 350,
            },
        ),
        (
            "C:/offline-pipeline/src/data_check",
            "main",
            "local",
            {
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": 0.2,
                "min_price": 10,
                "max_price": 350,
            },
        ),
    ]
