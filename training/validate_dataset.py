"""Validate an MVTec AD directory before attempting a training run."""

from __future__ import annotations

import argparse
import json

from visionguard.data import DatasetValidationError, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="mvtec_anomaly_detection")
    parser.add_argument("--category", action="append", help="Category to validate; repeat for more than one")
    parser.add_argument("--skip-image-verification", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_dataset(
            args.dataset_root,
            args.category,
            verify_images=not args.skip_image_verification,
            strict=True,
        )
    except DatasetValidationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
