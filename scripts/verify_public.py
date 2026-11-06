"""Verify the served model versions, examples and error handling over HTTP."""

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def timed(base: str, path: str, data: bytes | None = None) -> tuple[bytes, float]:
    start = perf_counter()
    headers = {"Content-Type": "application/octet-stream"} if data is not None else {}
    with urlopen(Request(base + path, data=data, headers=headers), timeout=90) as response:
        if urlsplit(response.geturl()).netloc != urlsplit(base).netloc:
            raise ValueError("Unexpected redirect outside the app host")
        result = response.read()
    return result, round((perf_counter() - start) * 1000, 2)


def verify_examples(
    base: str, category: str, version: str, preview: str, report: dict | None
) -> dict:
    results = {}
    endpoint = f"/api/predict?category={category}"
    if preview == "jpeg":
        endpoint += "&preview=jpeg"
    for kind in ("normal", "defect"):
        example, _ = timed(base, f"/api/examples/{category}/{kind}")
        payload, latency = timed(base, endpoint, example)
        result = json.loads(payload)
        if not all(
            result[key].startswith(f"data:image/{preview};base64,")
            for key in ("original", "overlay")
        ):
            raise ValueError(f"Missing comparison for {category}/{kind}")
        if result["category"] != category or result["model_version"] != version:
            raise ValueError(f"Unexpected serving artifact for {category}/{kind}")
        score, threshold = result["anomaly_score"], result["threshold"]
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise ValueError(f"Non-finite prediction for {category}/{kind}")
        if result["label"] != ("ANOMALY" if score > threshold else "NORMAL"):
            raise ValueError(f"Decision does not match threshold for {category}/{kind}")
        if report is not None:
            sample = report["samples"][kind]
            if hashlib.sha256(example).hexdigest() != sample["sha256"]:
                raise ValueError(f"Unexpected example bytes for {category}/{kind}")
            expected = next(row for row in report["predictions"] if row["image"] == sample["image"])
            if (
                not math.isclose(score, expected["score"], abs_tol=0.002, rel_tol=0.0001)
                or not math.isclose(threshold, report["threshold"], abs_tol=0.000001)
                or result["label"] != expected["label"]
            ):
                raise ValueError(f"Serving differs from the evaluated model for {category}/{kind}")
        results[kind] = {
            "label": result["label"], "score": score, "threshold": threshold, "http_ms": latency
        }
    return results


def verify(base: str, *, all_categories: bool = False, reports: Path | None = None) -> dict:
    page, first_ms = timed(base, "/")
    if b"VisionGuard" not in page:
        raise ValueError("The anonymous page did not load")
    payload, ready_ms = timed(base, "/ready")
    ready = set(json.loads(payload)["categories"])
    if all_categories:
        config = tomllib.loads(Path("configs/categories.toml").read_text(encoding="utf-8"))
        expected = {slug: info["artifact"] for slug, info in config["categories"].items()}
        payload, _ = timed(base, "/api/categories")
        offered = {
            row["id"]: row["model_version"] for row in json.loads(payload)["categories"]
        }
        if ready != set(expected) or offered != expected:
            raise ValueError(f"Expected all configured models, but got {sorted(ready)}")
        results = {"first_page_ms": first_ms, "ready_ms": ready_ms, "categories": {}}
        for slug, version in expected.items():
            report = (
                json.loads((reports / f"{slug}-evaluation.json").read_text(encoding="utf-8"))
                if reports is not None else None
            )
            results["categories"][slug] = verify_examples(base, slug, version, "jpeg", report)
    else:
        if "bottle" not in ready:
            raise ValueError("Bottle is not ready")
        results = {
            "first_page_ms": first_ms,
            "examples": verify_examples(base, "bottle", "bottle-pc-v1", "png", None),
        }
    try:
        timed(base, "/api/predict?category=bottle", b"invalid image")
    except HTTPError as error:
        with error:
            if error.code != 422 or json.load(error)["code"] != "invalid_image":
                raise ValueError("Unexpected invalid-upload response") from error
    else:
        raise ValueError("Invalid image was accepted")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument("--reports", type=Path, help="Cross-check all samples against evaluation reports")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    parts = urlsplit(base)
    loopback = parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost"}
    if (
        not (parts.scheme == "https" or loopback) or not parts.hostname or parts.path
        or parts.query or parts.fragment or parts.username
    ):
        parser.error("Provide an HTTPS app origin or a local loopback URL.")
    if args.reports and not args.all_categories:
        parser.error("--reports requires --all-categories.")
    print(json.dumps(verify(base, all_categories=args.all_categories, reports=args.reports), indent=2))


if __name__ == "__main__":
    main()
