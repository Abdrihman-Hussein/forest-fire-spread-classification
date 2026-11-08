const form = document.getElementById("predict-form");
const submitButton = document.getElementById("submit-button");
const submitLabel = submitButton.querySelector(".submit-label");
const errorBox = document.getElementById("form-error");

const amountEl = document.getElementById("result-amount");
const noteEl = document.getElementById("result-note");
const resultBody = document.getElementById("result-body");
const compareEl = document.getElementById("compare");
const compareNote = document.getElementById("compare-note");
const liveRegion = document.getElementById("live-region");

const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");

const prefersReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const percent = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const formatPercent = (value) => `${percent.format(value * 100)}%`;

// Each slider has a matching <output> that shows the live value.
const sliders = [
  { id: "DMC", decimals: 1, suffix: "" },
  { id: "DC", decimals: 1, suffix: "" },
  { id: "temp", decimals: 1, suffix: " °C" },
  { id: "RH", decimals: 0, suffix: "%" },
  { id: "wind", decimals: 1, suffix: " km/h" },
];

const sliderInputs = sliders.map((slider) => {
  const input = document.getElementById(slider.id);
  const output = document.getElementById(`${slider.id}-value`);

  const sync = () => {
    const value = Number(input.value);
    output.textContent = `${value.toFixed(slider.decimals)}${slider.suffix}`;
  };

  input.addEventListener("input", sync);
  sync();

  return { input, sync };
});

form.addEventListener("input", () => {
  errorBox.hidden = true;
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function animateAmount(target) {
  amountEl.classList.remove("is-placeholder");

  if (prefersReducedMotion() || target <= 0) {
    amountEl.textContent = formatPercent(target);
    return;
  }

  const duration = 650;
  const started = performance.now();

  const frame = (now) => {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    amountEl.textContent = formatPercent(target * eased);
    if (progress < 1) {
      requestAnimationFrame(frame);
    } else {
      amountEl.textContent = formatPercent(target);
    }
  };

  requestAnimationFrame(frame);
}

function renderCompare(data) {
  const rows = [
    { label: "Random forest", value: data.probability },
    { label: "Logistic regression", value: data.logistic_probability },
  ];

  const base = Number.isFinite(data.base_rate) ? data.base_rate : 0;

  compareEl.replaceChildren(
    ...rows.map((row) => {
      const li = document.createElement("li");
      li.className = "compare-row";

      const head = document.createElement("div");
      head.className = "compare-head";

      const label = document.createElement("span");
      label.className = "compare-label";
      label.textContent = row.label;

      const value = document.createElement("span");
      value.className = "compare-value";
      value.textContent = formatPercent(row.value);

      head.append(label, value);

      const track = document.createElement("div");
      track.className = "compare-track";

      const fill = document.createElement("span");
      fill.className = "compare-fill";
      track.append(fill);

      if (base > 0) {
        const marker = document.createElement("span");
        marker.className = "compare-base";
        marker.style.left = `${Math.min(100, base * 100)}%`;
        marker.title = `Base rate ${formatPercent(base)}`;
        track.append(marker);
      }

      li.append(head, track);

      requestAnimationFrame(() => {
        fill.style.width = `${Math.min(100, Math.max(0, row.value * 100))}%`;
      });

      return li;
    })
  );

  compareNote.textContent =
    `The dashed line marks the historical base rate of ${formatPercent(base)} — ` +
    `the share of fires that did spread. Anything near it is close to a coin flip.`;
}

function showResult(data) {
  animateAmount(data.probability);
  noteEl.textContent =
    `${data.label}. ${formatPercent(data.probability)} chance this fire burns area.`;
  resultBody.hidden = false;
  renderCompare(data);
  liveRegion.textContent =
    `${data.label}: ${formatPercent(data.probability)} chance of spread, ` +
    `against a base rate of ${formatPercent(data.base_rate)}.`;

  if (window.matchMedia("(max-width: 920px)").matches) {
    document.querySelector(".result-card").scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "start",
    });
  }
}

function readPayload() {
  const data = new FormData(form);

  return {
    X: Number(data.get("X")),
    Y: Number(data.get("Y")),
    DMC: Number(data.get("DMC")),
    DC: Number(data.get("DC")),
    temp: Number(data.get("temp")),
    RH: Number(data.get("RH")),
    wind: Number(data.get("wind")),
  };
}

// Mirrors the slider ranges in index.html. The server accepts the full
// dataset range, but the page only offers the 1st-99th percentile so no
// part of a slider is dead space.
const limits = {
  X: [1, 9],
  Y: [2, 9],
  DMC: [3.6, 286.9],
  DC: [16.2, 848.7],
  temp: [4.6, 32.1],
  RH: [19, 90],
  wind: [0.9, 8.5],
};

function validate(payload) {
  for (const [key, [low, high]] of Object.entries(limits)) {
    const value = payload[key];
    if (!Number.isFinite(value)) {
      return `${key} must be a number.`;
    }
    if (value < low || value > high) {
      return `${key} must be between ${low} and ${high}.`;
    }
  }
  return null;
}

// ---------- Park grid picker ----------
// The grid is built from /api/model-info, so the selectable cells are
// exactly the (X, Y) pairs that appear in the dataset. Y=7 has no fires
// at all, so that entire row renders disabled.

const gridEl = document.getElementById("park-grid");
const gridReadout = document.getElementById("grid-readout");
const xInput = document.getElementById("X");
const yInput = document.getElementById("Y");

let cellCounts = new Map();
let selected = { x: Number(xInput.value), y: Number(yInput.value) };

function updateReadout() {
  const n = cellCounts.get(`${selected.x},${selected.y}`) ?? 0;
  gridReadout.textContent =
    `X=${selected.x}, Y=${selected.y} · ${n} recorded fire${n === 1 ? "" : "s"}`;
}

function selectCell(x, y) {
  selected = { x, y };
  xInput.value = String(x);
  yInput.value = String(y);

  for (const cell of gridEl.querySelectorAll(".grid-cell")) {
    const on = Number(cell.dataset.x) === x && Number(cell.dataset.y) === y;
    cell.classList.toggle("is-selected", on);
    cell.setAttribute("aria-checked", on ? "true" : "false");
    cell.tabIndex = on ? 0 : -1;
  }

  updateReadout();
}

// Arrow keys move through the selectable cells like a radiogroup.
function onGridKeydown(event) {
  const moves = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, 1],
    ArrowDown: [0, -1],
  };
  const move = moves[event.key];
  if (!move) return;

  event.preventDefault();

  const x = Number(event.currentTarget.dataset.x) + move[0];
  const y = Number(event.currentTarget.dataset.y) + move[1];
  const next = gridEl.querySelector(
    `.grid-cell[data-x="${x}"][data-y="${y}"]:not(:disabled)`
  );

  if (next) {
    next.focus();
    selectCell(x, y);
  }
}

function renderGrid(grid) {
  if (!grid || !Array.isArray(grid.cells)) return;

  cellCounts = new Map(grid.cells.map((cell) => [`${cell.x},${cell.y}`, cell.n]));

  const cells = [];

  for (let y = grid.y_max; y >= grid.y_min; y -= 1) {
    for (let x = grid.x_min; x <= grid.x_max; x += 1) {
      const count = cellCounts.get(`${x},${y}`) ?? 0;

      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "grid-cell";
      cell.dataset.x = String(x);
      cell.dataset.y = String(y);
      cell.setAttribute("role", "radio");
      cell.setAttribute("aria-checked", "false");
      cell.tabIndex = -1;

      if (count === 0) {
        cell.disabled = true;
        cell.classList.add("is-empty");
        cell.setAttribute("aria-label", `X ${x}, Y ${y}: no recorded fires`);
        cell.title = `X ${x}, Y ${y} — no recorded fires`;
      } else {
        cell.setAttribute("aria-label", `X ${x}, Y ${y}: ${count} recorded fires`);
        cell.title = `X ${x}, Y ${y} — ${count} recorded fires`;
        cell.addEventListener("click", () => selectCell(x, y));
        cell.addEventListener("keydown", onGridKeydown);
      }

      cells.push(cell);
    }
  }

  gridEl.replaceChildren(...cells);

  // Keep the current selection if it exists, otherwise take the first
  // selectable cell so the form never starts on an impossible position.
  const current = gridEl.querySelector(
    `.grid-cell[data-x="${selected.x}"][data-y="${selected.y}"]:not(:disabled)`
  );
  const fallback = gridEl.querySelector(".grid-cell:not(:disabled)");
  const start = current ?? fallback;

  if (start) {
    selectCell(Number(start.dataset.x), Number(start.dataset.y));
  }
}

function gridUnavailable() {
  const note = document.createElement("p");
  note.className = "grid-fallback";
  note.textContent = "Map grid unavailable while the service is offline.";
  gridEl.replaceChildren(note);
  gridReadout.textContent = "X=?, Y=?";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;

  const payload = readPayload();
  const problem = validate(payload);
  if (problem) {
    showError(problem);
    return;
  }

  submitButton.disabled = true;
  submitLabel.textContent = "Estimating…";
  form.setAttribute("aria-busy", "true");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let message = "Something went wrong while estimating. Please try again.";
      try {
        const body = await response.json();
        if (typeof body.detail === "string") {
          message = body.detail;
        }
      } catch {
        /* keep the friendly default */
      }
      throw new Error(message);
    }

    showResult(await response.json());
  } catch (error) {
    showError(
      error instanceof TypeError
        ? "Could not reach the prediction service. Make sure the server is running."
        : error.message
    );
  } finally {
    submitButton.disabled = false;
    submitLabel.textContent = "Estimate spread risk";
    form.removeAttribute("aria-busy");
  }
});

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) throw new Error("Model info unavailable");
    const info = await response.json();

    if (Number.isFinite(info.accuracy)) {
      document.getElementById("metric-accuracy").textContent = info.accuracy.toFixed(3);
    }
    if (Number.isFinite(info.f1)) {
      document.getElementById("metric-f1").textContent = info.f1.toFixed(3);
    }
    if (Number.isFinite(info.roc_auc)) {
      document.getElementById("metric-auc").textContent = info.roc_auc.toFixed(3);
    }

    if (Number.isFinite(info.trained_samples) && Number.isFinite(info.features)) {
      document.getElementById("model-note").textContent =
        `Random forest · ${Number(info.trained_samples).toLocaleString()} training rows · ` +
        `${info.features} features · baseline ${formatPercent(info.baseline_accuracy)}.`;
    }

    if (Number.isFinite(info.dataset_rows)) {
      statusText.textContent = `v1 · ${Number(info.dataset_rows).toLocaleString()} fires`;
    }

    renderGrid(info.grid);
  } catch {
    statusPill.classList.add("is-offline");
    statusText.textContent = "Model info unavailable";
    gridUnavailable();
  }
}

sliderInputs.forEach((slider) => slider.sync());
loadModelInfo();
