#!/usr/bin/env python
"""
Download from W;&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    input_artifact = run.use_artifact(args.input_artifact)
    artifact_dir = Path(input_artifact.download())
    csv_files = sorted(artifact_dir.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in artifact {args.input_artifact}"
        )

    artifact_local_path = csv_files[0]
    logger.info("Reading raw data from %s", artifact_local_path)
    df = pd.read_csv(artifact_local_path)

    logger.info(
        "Filtering prices between %.2f and %.2f",
        args.min_price,
        args.max_price,
    )
    df = df[df["price"].between(args.min_price, args.max_price)].copy()
    df["last_review"] = pd.to_datetime(df["last_review"])

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for filtering the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for filtering the dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
