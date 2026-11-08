"""Smoke tests for the web layer.

Run with either:
    python -m unittest discover -s tests
    python -m pytest
"""

import json
import threading
import unittest
import urllib.error
import urllib.request

import app

SAMPLE = {
    "X": 7,
    "Y": 5,
    "DMC": 120,
    "DC": 600,
    "temp": 30,
    "RH": 25,
    "wind": 5,
}


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = app.create_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def get_json(self, path):
        status, body, _ = self.request("GET", path)
        return status, json.loads(body.decode("utf-8"))

    def post_json(self, path, payload):
        status, body, _ = self.request("POST", path, payload)
        return status, json.loads(body.decode("utf-8"))

    def test_health(self):
        status, body = self.get_json("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_index_serves_the_estimator_ui(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("Emberline", text)
        self.assertIn("/static/app.js", text)
        self.assertIn("/static/assets/forest-fire.jpg", text)
        self.assertIn("park-grid", text)
        self.assertIn("text/html", headers["Content-Type"])

    def test_index_no_longer_asks_for_dropped_features(self):
        # The lean rebuild removed these inputs; the page must not ask for
        # them again or the UI and the model would drift apart.
        _, body, _ = self.request("GET", "/")
        text = body.decode("utf-8")
        for removed in ("FFMC", "ISI", 'name="month"', 'name="rain"', 'name="day_type"'):
            self.assertNotIn(removed, text)

    def test_forest_fire_image_is_served(self):
        status, body, headers = self.request("GET", "/static/assets/forest-fire.jpg")
        self.assertEqual(status, 200)
        self.assertIn("image/jpeg", headers["Content-Type"])
        self.assertGreater(len(body), 10_000)
        self.assertEqual(body[:2], b"\xff\xd8")  # JPEG start-of-image marker

    def test_static_assets_are_served(self):
        status, body, headers = self.request("GET", "/static/styles.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers["Content-Type"])
        self.assertIn(b"--ember", body)

    def test_path_traversal_is_blocked(self):
        status, _, _ = self.request("GET", "/static/../app.py")
        self.assertEqual(status, 404)

    def test_prediction_matches_the_offline_pipeline(self):
        status, body = self.post_json("/api/predict", SAMPLE)
        self.assertEqual(status, 200)
        # Same values produced by predict.py for this profile.
        self.assertAlmostEqual(body["probability"], 0.5800, places=3)
        self.assertAlmostEqual(body["logistic_probability"], 0.6097, places=3)
        self.assertEqual(body["prediction"], 1)

    def test_prediction_includes_the_base_rate(self):
        _, body = self.post_json("/api/predict", SAMPLE)
        self.assertAlmostEqual(body["base_rate"], 0.5244, places=3)

    def test_dry_conditions_raise_the_probability(self):
        _, dry = self.post_json("/api/predict", SAMPLE)
        _, wet = self.post_json(
            "/api/predict",
            {**SAMPLE, "DMC": 5, "DC": 50, "temp": 5, "RH": 95, "wind": 1},
        )
        self.assertGreater(dry["probability"], wet["probability"])

    def test_rejects_a_grid_cell_with_no_recorded_fire(self):
        # Y=7 does not appear anywhere in the 513-row dataset.
        status, body = self.post_json("/api/predict", {**SAMPLE, "Y": 7})
        self.assertEqual(status, 422)
        self.assertIn("grid cell", body["detail"])

    def test_rejects_out_of_range_input(self):
        status, body = self.post_json("/api/predict", {**SAMPLE, "RH": 250})
        self.assertEqual(status, 422)
        self.assertIn("RH", body["detail"])

    def test_rejects_missing_field(self):
        payload = dict(SAMPLE)
        payload.pop("wind")
        status, body = self.post_json("/api/predict", payload)
        self.assertEqual(status, 422)
        self.assertIn("wind", body["detail"])

    def test_rejects_fractional_grid_coordinate(self):
        status, body = self.post_json("/api/predict", {**SAMPLE, "X": 7.5})
        self.assertEqual(status, 422)
        self.assertIn("whole number", body["detail"])

    def test_rejects_invalid_json(self):
        request = urllib.request.Request(
            self.base_url + "/api/predict",
            data=b"{not json",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=15)
            self.fail("Expected a 400 response")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 400)

    def test_model_info_reports_metrics(self):
        status, body = self.get_json("/api/model-info")
        self.assertEqual(status, 200)
        self.assertEqual(body["algorithm"], "Random Forest")
        self.assertEqual(body["features"], 7)
        self.assertEqual(
            body["feature_names"], ["X", "Y", "DMC", "DC", "temp", "RH", "wind"]
        )
        self.assertGreater(body["dataset_rows"], 0)
        self.assertGreater(body["trained_samples"], 0)
        self.assertGreater(body["roc_auc"], 0.5)
        self.assertLessEqual(body["roc_auc"], 1)
        self.assertAlmostEqual(body["baseline_accuracy"], 0.5244, places=3)

    def test_model_info_publishes_the_observed_grid(self):
        _, body = self.get_json("/api/model-info")
        grid = body["grid"]
        self.assertEqual(grid["x_min"], 1)
        self.assertEqual(grid["x_max"], 9)
        self.assertEqual(grid["y_min"], 2)
        self.assertEqual(grid["y_max"], 9)

        cells = {(cell["x"], cell["y"]) for cell in grid["cells"]}
        self.assertEqual(len(cells), 36)          # 36 of 72 cells occur
        self.assertNotIn((1, 7), cells)           # the empty Y=7 row
        self.assertIn((7, 5), cells)              # the default selection


if __name__ == "__main__":
    unittest.main()
