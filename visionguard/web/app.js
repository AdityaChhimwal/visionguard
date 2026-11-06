import { api } from "./api.js";

const $ = (id) => document.getElementById(id);
const category = $("category");
const input = $("image-file");
const analyze = $("analyze");
const output = $("output");
let categories = [];
let maxBytes = 0;
let image = null;
let previewUrl = null;
let active = null;
let busy = false;

function showError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function status(message) {
  $("status").textContent = message;
}
function controls() {
  input.disabled = !category.value;
  analyze.disabled = !image || !category.value || busy;
  analyze.textContent = busy ? "Working…" : "Analyze Image";
  output.setAttribute("aria-busy", String(busy));
}
function cancel() {
  active?.abort();
  active = null;
  busy = false;
}
function reset() {
  cancel();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  image = null;
  input.value = "";
  for (const id of ["preview", "original", "overlay"]) $(id).removeAttribute("src");
  $("preview-box").hidden = true;
  $("result").hidden = true;
  $("empty").hidden = false;
  $("preview-title").textContent = "Inspection preview";
  showError();
  status("");
  controls();
}

function selectImage(file) {
  if (!category.value || !file) return;
  reset();
  if (file.size === 0) return showError("This file is empty. Choose another image.");
  if (file.size > maxBytes) return showError("This file exceeds the upload limit.");
  if (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    return showError("Choose a JPEG, PNG, or WebP image.");
  }
  image = file;
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("filename").textContent = file.name;
  $("empty").hidden = true;
  $("preview-box").hidden = false;
  status("Image selected. Ready to analyze.");
  controls();
}

async function useSample(sample) {
  reset();
  const task = new AbortController();
  active = task;
  busy = true;
  status("Loading example…");
  controls();
  try {
    const response = await api(sample.url, { signal: task.signal });
    const blob = await response.blob();
    if (active !== task) return;
    selectImage(new File([blob], `${sample.name}.png`, { type: "image/png" }));
    analyze.focus();
  } catch (error) {
    if (active !== task) return;
    showError(error.message);
    status("");
  } finally {
    if (active === task) {
      active = null;
      busy = false;
      controls();
    }
  }
}

function chooseCategory() {
  reset();
  const entry = categories.find((item) => item.id === category.value);
  $("category-hint").textContent = entry?.description ?? "Select a category that matches your product.";
  $("examples").hidden = !entry?.samples.length;
  $("example-buttons").replaceChildren(...(entry?.samples ?? []).map((sample) => {
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = sample.name;
    button.addEventListener("click", () => useSample(sample));
    return button;
  }));
}

async function discover() {
  $("retry").hidden = true;
  showError();
  status("Checking model availability…");
  try {
    const result = await (await api("/api/categories", { timeout: 20000 })).json();
    categories = result.categories;
    maxBytes = result.max_upload_bytes;
    $("file-limit").textContent = `or drop a JPEG, PNG, or WebP (up to ${(maxBytes / 1048576).toFixed(1)} MiB)`;
    category.replaceChildren(...categories.map((entry) => new Option(entry.name, entry.id)));
    if (!categories.length) throw new Error("No inspection model is ready. Try again later.");
    category.disabled = false;
    chooseCategory();
  } catch (error) {
    category.replaceChildren(new Option("No category available", ""));
    category.disabled = true;
    chooseCategory();
    $("retry").hidden = false;
    showError(error.message);
  }
}

function render(result) {
  const anomaly = result.label === "ANOMALY";
  $("label").textContent = result.label;
  $("label").classList.toggle("anomaly", anomaly);
  $("verdict-text").textContent = anomaly ? "The score exceeds the threshold." : "The score does not exceed the threshold.";
  $("score").textContent = result.anomaly_score.toPrecision(7);
  $("threshold").textContent = result.threshold.toPrecision(7);
  $("original").src = result.original;
  $("overlay").src = result.overlay;
  $("scale").textContent = `Warmer areas show higher patch distances. The display scale is fixed from 0 to ${result.heatmap_range[1].toFixed(3)} for this model.`;
  $("runtime").textContent = `${result.model_version} · ${Math.round(result.processing_ms)} ms server processing`;
  $("preview-title").textContent = "Inspection result";
  $("preview-box").hidden = true;
  $("result").hidden = false;
  $("preview-title").focus();
}

analyze.addEventListener("click", async () => {
  if (!image || !category.value || busy) return;
  const task = new AbortController();
  active = task;
  busy = true;
  showError();
  status("Analyzing image…");
  controls();
  try {
    const url = `/api/predict?category=${encodeURIComponent(category.value)}`;
    const response = await api(url, {
      method: "POST", body: image, signal: task.signal,
      headers: { "Content-Type": "application/octet-stream" },
    });
    const result = await response.json();
    if (active !== task) return;
    render(result);
    status("Inspection complete.");
  } catch (error) {
    if (active !== task) return;
    showError(error.message);
    status("");
  } finally {
    if (active === task) {
      active = null;
      busy = false;
      controls();
    }
  }
});

input.addEventListener("change", () => selectImage(input.files[0]));
category.addEventListener("change", chooseCategory);
$("reset").addEventListener("click", () => { reset(); input.focus(); });
$("retry").addEventListener("click", discover);
const drop = $("drop");
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (category.value) drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("drag");
  selectImage(event.dataTransfer.files[0]);
});
discover();
