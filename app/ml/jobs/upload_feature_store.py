#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from app.db.models.feature_store import FeatureStore


def main():
    parser = argparse.ArgumentParser(
        description="Upload transformed feature arrays into the Feature Store (PostgreSQL via Neon)."
    )
    parser.add_argument(
        "--data-path",
        default="data/model_artifacts",
        help="Directory containing X_train.npy, X_val.npy, X_test.npy, y_*.npy files",
    )
    parser.add_argument(
        "--version",
        default=os.getenv("MODEL_VERSION", "v1"),
        help="Version tag for these features (default: MODEL_VERSION env or 'v1')",
    )
    args = parser.parse_args()

    neon_url = os.getenv("NEON_URL_PROD")
    if not neon_url:
        print("Error: NEON_URL_PROD environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(neon_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    for split in ["X_train", "X_val", "X_test"]:
        npy_file = os.path.join(args.data_path, f"{split}.npy")
        if not os.path.exists(npy_file):
            print(f"Skipping {split}: file not found at {npy_file}", file=sys.stderr)
            continue
        features_array = np.load(npy_file)
        print(f"Inserting {features_array.shape[0]} rows from {split} ...")
        for row in tqdm(features_array.tolist(), desc=f"Inserting {split}"):
            fs_record = FeatureStore(features=row, version=args.version)
            session.add(fs_record)
        session.commit()

    session.close()
    print("Feature upload to Neon database complete.")


if __name__ == "__main__":
    main()
