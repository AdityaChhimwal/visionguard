"""Verify the full MVTec AD picker and real inspection flows in Chromium."""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import Page, expect, sync_playwright

from scripts.verify_browser import assert_layout

MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
}
SAMPLE_BUTTONS = {"normal": "Normal example", "defect": "Defective example"}


def open_catalog(page: Page, url: str) -> None:
    page.goto(url, timeout=90_000)
    category = page.locator("#category")
    expect(category).to_be_enabled(timeout=90_000)
    assert set(category.locator("option").evaluate_all("(items) => items.map(item => item.value)")) == (
        MVTEC_CATEGORIES
    )
    expect(category.locator('optgroup[label="Objects"] option')).to_have_count(10)
    expect(category.locator('optgroup[label="Textures"] option')).to_have_count(5)
    expect(page.locator("#category-count")).to_have_text("15 categories")


def select_category(page: Page, slug: str) -> None:
    page.locator("#category").select_option(slug)
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("#preview-box")).to_be_hidden()
    expect(page.get_by_role("button", name="Analyze Image", exact=True)).to_be_disabled()


def inspect(page: Page, url: str, slug: str, kind: str, *, upload: bool = False) -> dict:
    picker = page.get_by_label("Choose an image")
    if upload:
        sample = page.request.get(f"{url}/api/examples/{slug}/{kind}")
        assert sample.ok and sample.headers["content-type"] == "image/png"
        picker.set_input_files({
            "name": f"{slug}-{kind}.png", "mimeType": "image/png", "buffer": sample.body()
        })
        assert picker.evaluate("(input) => input.files.length") == 1
    else:
        page.get_by_role("button", name=SAMPLE_BUTTONS[kind], exact=True).click()
    expect(page.locator("#preview")).to_be_visible(timeout=60_000)

    def prediction_for_category(response):
        target = urlsplit(response.url)
        return (
            target.path == "/api/predict"
            and parse_qs(target.query) == {"category": [slug], "preview": ["jpeg"]}
            and response.request.method == "POST"
        )

    with page.expect_response(prediction_for_category, timeout=60_000) as response:
        page.get_by_role("button", name="Analyze Image", exact=True).click()
    prediction = response.value
    assert prediction.ok, prediction.text()
    result = prediction.json()
    assert result["category"] == slug
    assert math.isfinite(result["anomaly_score"]) and math.isfinite(result["threshold"])
    decision = "ANOMALY" if result["anomaly_score"] > result["threshold"] else "NORMAL"
    assert result["label"] == decision

    expect(page.locator("#result")).to_be_visible(timeout=60_000)
    expect(page.locator("#label")).to_have_text(decision)
    category_name = page.locator("#category option:checked").text_content()
    expect(page.locator("#result-category")).to_have_text(category_name)
    page.wait_for_function(
        """() => ['original', 'overlay'].every(id => {
          const image = document.getElementById(id);
          return image.src.startsWith('data:image/jpeg;base64,') && image.naturalWidth > 0;
        })"""
    )
    assert_layout(page)
    return {
        "category": slug,
        "example": kind,
        "label": decision,
        "expected": "NORMAL" if kind == "normal" else "ANOMALY",
        "score": result["anomaly_score"],
        "threshold": result["threshold"],
        "model_version": result["model_version"],
        "processing_ms": result["processing_ms"],
    }


def reset(page: Page) -> None:
    page.get_by_role("button", name="Analyze Another Image").click()
    expect(page.locator("#result")).to_be_hidden()
    expect(page.get_by_label("Choose an image")).to_be_focused()


def verify(url: str, output: Path, channel: str | None) -> dict:
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel)
        desktop = browser.new_page(viewport={"width": 1280, "height": 880})
        desktop.on("pageerror", lambda error: errors.append(str(error)))
        open_catalog(desktop, url)
        desktop_results = []
        for slug in ("bottle", "cable", "carpet"):
            select_category(desktop, slug)
            desktop_results.append(inspect(desktop, url, slug, "normal", upload=slug == "bottle"))
            reset(desktop)
            desktop_results.append(inspect(desktop, url, slug, "defect"))
        desktop.screenshot(path=output / "desktop.png", full_page=True, animations="disabled")
        reset(desktop)
        desktop.get_by_label("Choose an image").set_input_files({
            "name": "invalid.png", "mimeType": "image/png", "buffer": b"not an image"
        })
        desktop.get_by_role("button", name="Analyze Image", exact=True).click()
        expect(desktop.get_by_role("alert")).to_contain_text("valid JPEG, PNG, or WebP")
        expect(desktop.locator("#result")).to_be_hidden()

        mobile = browser.new_page(**playwright.devices["iPhone 13"])
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        open_catalog(mobile, url)
        select_category(mobile, "carpet")
        mobile_result = inspect(mobile, url, "carpet", "defect")
        mobile.screenshot(path=output / "mobile.png", full_page=True, animations="disabled")
        browser.close()
    assert not errors, errors
    return {
        "available_categories": len(MVTEC_CATEGORIES),
        "desktop": desktop_results,
        "mobile": mobile_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL of a running VisionGuard service")
    parser.add_argument("--output", type=Path, default=Path("outputs/mvtec-browser"))
    parser.add_argument("--channel", help="Use an installed Chromium channel, e.g. chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = json.dumps(verify(args.url.rstrip("/"), args.output, args.channel), indent=2)
    (args.output / "browser-results.json").write_text(results + "\n", encoding="utf-8")
    print(results)


if __name__ == "__main__":
    main()
