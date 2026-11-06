"""Exercise the browser against a running, prepared Bottle model."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

from scripts.benchmark import server
from scripts.stage_examples import EXAMPLES


def assert_layout(page: Page) -> None:
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for name in ("Original inspection image", "Same image with warmer colors"):
        expect(page.get_by_alt_text(name, exact=False)).to_be_visible()
    assert page.locator("#original").evaluate("(img) => img.naturalWidth > 0")
    assert page.locator("#overlay").evaluate("(img) => img.naturalWidth > 0")


def flow(page: Page, url: str, dataset: Path, screenshot: Path, mobile: bool) -> str:
    js_errors = []
    page.on("pageerror", lambda error: js_errors.append(str(error)))
    page.goto(url)
    expect(page.locator("#category")).to_have_value("bottle")
    page.evaluate("""() => {
      window.inspectionStates = [];
      const status = document.querySelector('#status');
      new MutationObserver(() => window.inspectionStates.push(status.textContent))
        .observe(status, { childList: true, characterData: true, subtree: true });
    }""")
    if mobile:
        page.get_by_role("button", name="Defective example").click()
    else:
        page.get_by_label("Choose an image").set_input_files(str(dataset / EXAMPLES["normal"]))
    expect(page.locator("#preview")).to_be_visible()
    page.get_by_role("button", name="Analyze Image", exact=True).click()
    expect(page.locator("#result")).to_be_visible(timeout=30_000)
    assert "Analyzing image…" in page.evaluate("window.inspectionStates")
    label = page.locator("#label").inner_text()
    assert label in ("NORMAL", "ANOMALY")
    score = float(page.locator("#score").inner_text())
    threshold = float(page.locator("#threshold").inner_text())
    assert (score > threshold) == (label == "ANOMALY")
    assert_layout(page)
    page.screenshot(path=screenshot, full_page=True, animations="disabled")
    page.get_by_role("button", name="Analyze Another Image").click()
    expect(page.locator("#result")).to_be_hidden()
    expect(page.get_by_label("Choose an image")).to_be_focused()
    if not mobile:
        # Drag-and-drop an invalid PNG and verify the real HTTP decoding error.
        transfer = page.evaluate_handle("""() => {
          const dt = new DataTransfer();
          dt.items.add(new File(['garbage'], 'bad.png', {type: 'image/png'}));
          return dt;
        }""")
        page.locator("#drop").dispatch_event("drop", {"dataTransfer": transfer})
        page.get_by_role("button", name="Analyze Image", exact=True).click()
        expect(page.get_by_role("alert")).to_contain_text("valid JPEG, PNG, or WebP")
        expect(page.locator("#result")).to_be_hidden()
    assert not js_errors, js_errors
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--channel", help="Use an installed Chromium channel, e.g. chrome")
    args = parser.parse_args()
    with server(args.output / "browser-serve.log") as (url, _), sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel)
        desktop = browser.new_page(viewport={"width": 1280, "height": 880})
        mobile = browser.new_page(**playwright.devices["iPhone 13"])
        results = {
            "desktop_normal_upload": flow(desktop, url, args.dataset, args.output / "desktop.png", False),
            "mobile_defect_example": flow(mobile, url, args.dataset, args.output / "mobile.png", True),
        }
        browser.close()
    (args.output / "browser-results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
