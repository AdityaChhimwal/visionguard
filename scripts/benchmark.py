"""Start a fresh single-worker server and measure its real HTTP inference path."""

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Event, Thread

import psutil

from scripts.stage_examples import EXAMPLES


def request(url: str, data: bytes | None = None) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20) as reply:
        return json.load(reply)


@contextmanager
def server(log: Path) -> Iterator[tuple[str, dict]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "visionguard.api.main:app", "--host",
             "127.0.0.1", "--port", str(port), "--workers", "1", "--no-access-log"],
            stdout=stream, stderr=subprocess.STDOUT,
        )
        process = psutil.Process(proc.pid)
        samples = []
        stop = Event()

        def monitor():
            while not stop.is_set():
                try:
                    group = [process, *process.children(recursive=True)]
                    samples.append(sum(child.memory_info().rss for child in group))
                except psutil.Error:
                    return
                stop.wait(0.01)

        thread = Thread(target=monitor, daemon=True)
        thread.start()
        evidence = {}
        try:
            while True:
                try:
                    ready = request(url + "/ready")
                    break
                except (OSError, TimeoutError) as exc:
                    if proc.poll() is not None or time.perf_counter() - start > 25:
                        raise RuntimeError(f"Serving did not become ready. See {log}") from exc
                    time.sleep(0.05)
            evidence.update(
                startup_seconds=round(time.perf_counter() - start, 3),
                readiness=ready,
                os=platform.platform(),
                cpu=platform.processor() or platform.machine(),
                logical_cpus=os.cpu_count(),
            )
            yield url, evidence
        finally:
            stop.set()
            thread.join()
            evidence["peak_rss_mib"] = round(max(samples, default=0) / 1024 ** 2, 2)
            # Windows virtualenv launchers can spawn the actual server process.
            for child in reversed(process.children(recursive=True)):
                with suppress(psutil.NoSuchProcess):
                    child.terminate()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def measure(url: str, dataset: Path) -> dict:
    data = {kind: (dataset / name).read_bytes() for kind, name in EXAMPLES.items()}
    endpoint = url + "/api/predict?category=bottle"
    examples = {}
    timings = []
    for kind, payload in data.items():
        result = request(endpoint, payload)
        assert result["overlay"].startswith("data:image/png;base64,")
        examples[kind] = {
            key: result[key] for key in ("label", "anomaly_score", "threshold", "model_version")
        }
    for _ in range(8):
        for payload in data.values():
            start = time.perf_counter()
            request(endpoint, payload)
            timings.append((time.perf_counter() - start) * 1000)
    try:
        request(endpoint, b"this is not an image")
    except urllib.error.HTTPError as error:
        with error:
            assert error.code == 422 and json.load(error)["code"] == "invalid_image"
    else:
        raise AssertionError("Malformed upload was accepted")
    return {
        "examples": examples,
        "warm_requests": len(timings),
        "median_http_ms": round(statistics.median(timings), 2),
        "p95_http_ms": round(statistics.quantiles(timings, n=20, method="inclusive")[18], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    with server(args.output / "serve.log") as (url, evidence):
        evidence.update(measure(url, args.dataset))
    evidence["artifact_bytes"] = sum(
        path.stat().st_size for path in Path("artifacts/bottle-pc-v1").iterdir() if path.is_file()
    )
    (args.output / "bottle-runtime.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
