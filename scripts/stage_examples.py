"""Copy two fixed, attributed Bottle examples into the optional sample directory."""

import argparse
from pathlib import Path
from shutil import copyfile

EXAMPLES = {"normal": "test/good/000.png", "defect": "test/broken_large/000.png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/samples/bottle"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in EXAMPLES.items():
        copyfile(args.dataset / source, args.output / f"{name}.png")
    (args.output / "NOTICE.txt").write_text(
        "MVTec AD, Bottle category, copyright MVTec Software GmbH.\n"
        "https://www.mvtec.com/research-teaching/datasets/mvtec-ad\n"
        "CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        "Unchanged PNG bytes, renamed from test/good/000.png and test/broken_large/000.png.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
