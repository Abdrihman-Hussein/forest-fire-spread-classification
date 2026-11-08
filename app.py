"""Web layer for the Forest Fire Spread classifier.

A small, dependency-free HTTP server (Python standard library) that:
  * serves the single-page estimator in web/
  * exposes POST /api/predict and GET /api/model-info

Run with:  python app.py
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RF_PATH = BASE_DIR / "models" / "random_forest.pkl"
LR_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "dataset" / "clean_fire.csv"
WEB_DIR = BASE_DIR / "web"

# The documented range of every input column in the dataset. The server
# accepts anything the data actually contains; the page narrows its
# sliders to the 1st-99th percentile so they are not mostly dead space.
RANGES = {
    "X": (1.0, 9.0),
    "Y": (2.0, 9.0),
    "DMC": (1.1, 291.3),
    "DC": (7.9, 860.6),
    "temp": (2.2, 33.3),
    "RH": (15.0, 100.0),
    "wind": (0.4, 9.4),
}

WHOLE_NUMBERS = {"X", "Y", "RH"}

REQUIRED = [
    "X",
    "Y",
    "DMC",
    "DC",
    "temp",
    "RH",
    "wind",
]

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}

MAX_BODY_BYTES = 64_000

for path in (RF_PATH, LR_PATH, SCALER_PATH, COLUMNS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at {path}. Run `python train.py` first."
        )

rf_model = joblib.load(RF_PATH)
lr_model = joblib.load(LR_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = joblib.load(COLUMNS_PATH)

BASE_RATE = 0.5
GRID_CELLS: dict[tuple[int, int], int] = {}
if DATA_PATH.exists():
    _clean = pd.read_csv(DATA_PATH)
    BASE_RATE = float(_clean["fire_occurred"].mean())
    GRID_CELLS = {
        (int(x), int(y)): int(n)
        for (x, y), n in _clean.groupby(["X", "Y"]).size().items()
    }


class ValidationError(ValueError):
    """Raised when a prediction request is not usable."""


def validate_event(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValidationError("Request body must be a JSON object.")

    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        raise ValidationError(f"Missing field: {missing[0]}.")

    event: dict = {}

    for key, (low, high) in RANGES.items():
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            raise ValidationError(f"{key} must be a number.") from None

        if not low <= value <= high:
            raise ValidationError(f"{key} must be between {low} and {high}.")

        if key in WHOLE_NUMBERS:
            if value != int(value):
                raise ValidationError(f"{key} must be a whole number.")
            value = int(value)

        event[key] = value

    # Reject grid cells that never occur in the data. Y=7 is absent from
    # every one of the 513 recorded fires, so a prediction there would be
    # an extrapolation the model cannot support.
    if GRID_CELLS and (event["X"], event["Y"]) not in GRID_CELLS:
        raise ValidationError(
            f"No recorded fire at grid cell X={event['X']}, Y={event['Y']}."
        )

    return event


def encode_event(event: dict) -> dict:
    """Encode a validated event exactly like preprocessing.py / predict.py."""
    return {name: float(event[name]) for name in FEATURE_COLUMNS}


def predict(event: dict) -> dict:
    row = encode_event(event)
    frame = pd.DataFrame([[row[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    probability = float(rf_model.predict_proba(frame)[0][1])

    scaled = pd.DataFrame(scaler.transform(frame), columns=FEATURE_COLUMNS)
    logistic_probability = float(lr_model.predict_proba(scaled)[0][1])

    prediction = 1 if probability >= 0.5 else 0

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Likely to spread" if prediction else "Likely to burn out",
        "logistic_probability": round(logistic_probability, 4),
        "logistic_prediction": 1 if logistic_probability >= 0.5 else 0,
        "base_rate": round(BASE_RATE, 4),
    }


@lru_cache(maxsize=1)
def model_info() -> dict:
    """Recompute the same evaluation split shown in evaluate.py."""
    info: dict = {
        "algorithm": "Random Forest",
        "features": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "base_rate": round(BASE_RATE, 4),
        "grid": {
            "x_min": int(RANGES["X"][0]),
            "x_max": int(RANGES["X"][1]),
            "y_min": int(RANGES["Y"][0]),
            "y_max": int(RANGES["Y"][1]),
            "cells": [
                {"x": x, "y": y, "n": n}
                for (x, y), n in sorted(GRID_CELLS.items(), key=lambda item: (-item[0][1], item[0][0]))
            ],
        },
    }
    if not DATA_PATH.exists():
        return info

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["fire_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    predictions = rf_model.predict(X_test)
    probabilities = rf_model.predict_proba(X_test)[:, 1]

    info.update(
        {
            "dataset_rows": int(len(df)),
            "trained_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        }
    )
    return info


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "Emberline/1.0"
    protocol_version = "HTTP/1.1"

    def handle(self) -> None:
        # Browsers and PowerShell routinely drop keep-alive sockets; that is
        # not an application error, so do not dump a traceback for it.
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html")
            return

        if path.startswith("/static/"):
            self._send_static(path[len("/static/") :])
            return

        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if path == "/api/model-info":
            self._send_json(HTTPStatus.OK, model_info())
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found."})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Request body is required."})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"detail": "Request body is too large."},
            )
            return

        body = self.rfile.read(length)
        try:
            raw = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Request body must be valid JSON."})
            return

        try:
            event = validate_event(raw)
        except ValidationError as exc:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(exc)})
            return

        self._send_json(HTTPStatus.OK, predict(event))

    def _send_static(self, relative: str) -> None:
        target = (WEB_DIR / unquote(relative)).resolve()
        if WEB_DIR.resolve() not in target.parents or not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found."})
            return
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        try:
            payload = path.read_bytes()
        except OSError:
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found."})
            return

        content_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib naming
        print(f"[web] {self.address_string()} - {format % args}")


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), RequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Emberline estimator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"Emberline running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
