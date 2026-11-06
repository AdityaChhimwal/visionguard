"""Copy attributed MVTec AD examples into the optional sample directory."""

import argparse
from pathlib import Path
from shutil import copyfile

EXAMPLES = {"normal": "test/good/000.png", "defect": "test/broken_large/000.png"}


def stage_examples(dataset: Path, output: Path, category: str, examples: dict[str, str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, source in examples.items():
        copyfile(dataset / source, output / f"{name}.png")
    (output / "NOTICE.txt").write_text(
        f"MVTec AD, {category.replace('_', ' ').title()} category, copyright MVTec Software GmbH.\n"
        "https://www.mvtec.com/research-teaching/datasets/mvtec-ad\n"
        "CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        f"Unchanged PNG bytes, renamed from {examples['normal']} and {examples['defect']}.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/samples/bottle"))
    args = parser.parse_args()
    stage_examples(args.dataset, args.output, "bottle", EXAMPLES)


if __name__ == "__main__":
    main()
