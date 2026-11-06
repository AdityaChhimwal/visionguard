"""Check an anonymous HTTPS deployment with the two pinned Bottle examples."""

import argparse
import json
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def timed(base: str, path: str, data: bytes | None = None) -> tuple[bytes, float]:
    start = perf_counter()
    with urlopen(Request(base + path, data=data), timeout=90) as response:
        if urlsplit(response.geturl()).netloc != urlsplit(base).netloc:
            raise ValueError("Unexpected redirect outside the app host")
        result = response.read()
    return result, round((perf_counter() - start) * 1000, 2)


def verify(base: str) -> dict:
    page, first_ms = timed(base, "/")
    if b"Inspect a product image" not in page:
        raise ValueError("The anonymous page did not load")
    payload, _ = timed(base, "/ready")
    if "bottle" not in json.loads(payload)["categories"]:
        raise ValueError("Bottle is not ready")
    results = {"first_page_ms": first_ms, "examples": {}}
    endpoint = "/api/predict?category=bottle"
    for kind in ("normal", "defect"):
        example, _ = timed(base, f"/api/examples/bottle/{kind}")
        payload, latency = timed(base, endpoint, example)
        result = json.loads(payload)
        if not all(result[key].startswith("data:image/png;base64,") for key in ("original", "overlay")):
            raise ValueError("Missing PNG comparison")
        if result["model_version"] != "bottle-pc-v1":
            raise ValueError("Unexpected serving artifact")
        score, threshold = result["anomaly_score"], result["threshold"]
        if result["label"] != ("ANOMALY" if score > threshold else "NORMAL"):
            raise ValueError("Decision does not match score and threshold")
        results["examples"][kind] = {
            "label": result["label"], "score": score, "threshold": threshold, "http_ms": latency
        }
    try:
        timed(base, endpoint, b"invalid image")
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
    args = parser.parse_args()
    base = args.url.rstrip("/")
    parts = urlsplit(base)
    if parts.scheme != "https" or not parts.hostname or parts.path or parts.query or parts.fragment:
        parser.error("Provide an HTTPS app origin, e.g. https://your-service.onrender.com.")
    print(json.dumps(verify(base), indent=2))


if __name__ == "__main__":
    main()
