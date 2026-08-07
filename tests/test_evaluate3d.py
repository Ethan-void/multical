import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from multical.app.evaluate3d import evaluate_reconstruction


def test_evaluate3d_matches_frames_converts_units_and_checks_thresholds():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    reconstruction_path = root / "triangulation.json"
    reconstruction_path.write_text(json.dumps({
      "coordinate_frame": "world",
      "world_units": "meters",
      "frames": [
        {
          "frame": "P01",
          "status": "ok",
          "point_world": [1.01, 1.98, 3.02],
          "cameras_used": ["C1", "C2"],
          "reprojection_rms_px": 0.4
        },
        {
          "frame": "P02",
          "status": "ok",
          "point_world": [2.0, 1.0, 0.5],
          "cameras_used": ["C2", "C3"],
          "reprojection_rms_px": 0.3
        },
        {
          "frame": "P03",
          "status": "failed",
          "reason": "poor ray angle"
        },
        {
          "frame": "not-measured",
          "status": "ok",
          "point_world": [0.0, 0.0, 0.0]
        }
      ]
    }), encoding="utf-8")
    ground_truth_path = root / "measured.yaml"
    ground_truth_path.write_text(yaml.safe_dump({
      "coordinate_frame": "world",
      "world_units": "mm",
      "points": {
        "P01": [1000.0, 2000.0, 3000.0],
        "P02": [2000.0, 1000.0, 500.0],
        "P03": [0.0, 0.0, 0.0],
        "P04": [1000.0, 1000.0, 1000.0]
      }
    }), encoding="utf-8")

    result, destination, xlsx_destination = evaluate_reconstruction(
      reconstruction_path,
      ground_truth_path,
      max_mean_error=0.02,
      max_p95_error=0.02,
      max_error=0.04
    )

    assert destination.is_file()
    assert xlsx_destination.is_file()
    assert xlsx_destination.stat().st_size > 0
    assert result["summary"]["ground_truth_count"] == 4
    assert result["summary"]["matched_count"] == 2
    assert result["summary"]["failed_reconstruction_count"] == 1
    assert result["summary"]["missing_reconstruction_count"] == 1
    assert result["summary"]["unmeasured_reconstruction_count"] == 1
    assert np.allclose(
      result["points"][0]["error_xyz"], [0.01, -0.02, 0.02]
    )
    assert abs(result["points"][0]["error_3d"] - 0.03) < 1e-12
    assert result["acceptance"]["passed"] is False
    assert any(
      "no reconstruction" in failure
      for failure in result["acceptance"]["failures"]
    )
    assert any(
      "failed reconstruction" in failure
      for failure in result["acceptance"]["failures"]
    )
    assert any(
      "max_p95_error" in failure
      for failure in result["acceptance"]["failures"]
    )
    assert result["sources"]["ground_truth_scale_to_output_units"] == 0.001
