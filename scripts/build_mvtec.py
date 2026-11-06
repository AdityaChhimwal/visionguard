"""Fit and evaluate one full MVTec AD category for a release matrix."""

import argparse
import json
import sys
import tempfile
import tomllib
from pathlib import Path

from scripts.dataset import sha256
from scripts.evaluate import evaluate
from scripts.fetch_mvtec import CATEGORY_COUNTS, WEIGHTS_NAME, validate_inventory
from scripts.prepare import prepare
from scripts.stage_examples import stage_examples

PIXEL_BINS = 65536


def build(category: str, data: Path, artifacts: Path, reports: Path, config: Path) -> dict:
    dataset = data / "mvtec" / category
    validate_inventory(
        category, [f"{category}/{p.relative_to(dataset).as_posix()}" for p in dataset.rglob("*.png")]
    )
    version = f"{category}-pc-v1"
    settings = tomllib.loads(config.read_text(encoding="utf-8"))
    settings.update(category=category, version=version)
    model_output = artifacts / version
    examples_output = artifacts / "samples" / category
    report_output = reports / f"{category}-evaluation.json"
    for path in (model_output, examples_output, report_output):
        if path.exists():
            raise FileExistsError(f"{path} already exists. Use new output directories.")
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-mvtec-", dir=artifacts) as stage:
        staged_bundle = Path(stage) / version
        command = ["python", "-m", "scripts.build_mvtec", *sys.argv[1:]]
        prepare(settings, dataset, data / WEIGHTS_NAME, staged_bundle, command=command)
        report = evaluate(
            dataset, staged_bundle, category, version, settings["threads"], pixel_bins=PIXEL_BINS,
            command=command,
        )
        selected = {}
        for key, label in (("normal", "NORMAL"), ("defect", "ANOMALY")):
            selected[key] = next(
                (row for row in report["predictions"] if row["target"] == row["label"] == label),
                None,
            )
            if selected[key] is None:
                raise ValueError(f"{category} has no correctly classified {label} test example.")
        report["samples"] = {
            key: {"image": row["image"], "sha256": sha256(dataset / row["image"])}
            for key, row in selected.items()
        }
        report["example_selection"] = "First correctly classified image of each class after scoring"
        staged_examples = Path(stage) / "samples"
        stage_examples(
            dataset, staged_examples, category,
            {key: row["image"] for key, row in selected.items()},
        )
        examples_output.parent.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        with report_output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        staged_examples.rename(examples_output)
        staged_bundle.rename(model_output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORY_COUNTS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path("configs/bottle.toml"))
    args = parser.parse_args()
    report = build(args.category, args.data, args.artifacts, args.reports, args.config)
    print(json.dumps({k: v for k, v in report.items() if k not in {"predictions", "artifact"}}, indent=2))


if __name__ == "__main__":
    main()
