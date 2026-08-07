import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from multical.camera import Camera
from multical.io.export_calib import export_single
from multical.io.import_calib import load_calibration


def test_intrinsic_quality_is_exported_and_remains_load_compatible():
  camera = Camera(
    image_size=(640, 480),
    intrinsic=np.array([
      [700.0, 0.0, 320.0],
      [0.0, 701.0, 240.0],
      [0.0, 0.0, 1.0]
    ]),
    dist=np.zeros((1, 5)),
    error_perview=np.array([[0.20], [0.30], [0.40]]),
    intrinsic_dataset={
      "image_ids": [2, 4, 7],
      "board_ids": [0, 0, 1],
      "point_counts": [40, 38, 35],
      "detected_view_count": 5,
      "candidate_view_count": 4,
      "detected_image_count": 5,
      "candidate_image_count": 4
    }
  )

  with TemporaryDirectory() as temporary:
    output = Path(temporary) / "intrinsic.json"
    export_single(
      output,
      [camera],
      ["Cam1"],
      [[
        "Cam1/0000.jpg",
        "Cam1/0001.jpg",
        "Cam1/0002.jpg",
        "Cam1/0004.jpg",
        "Cam1/0007.jpg",
        "Cam1/0008.jpg"
      ]],
      errors=[0.25]
    )

    exported = json.loads(output.read_text(encoding="utf-8"))
    quality = exported["cameras"]["Cam1"]["quality"]
    assert quality["RMS"] == 0.25
    assert abs(quality["mean_view_RMS"] - 0.30) < 1e-12
    assert quality["max_view_RMS"] == 0.40
    assert quality["view_count"] == 3
    assert quality["image_count"] == 3
    assert quality["input_image_count"] == 6
    assert quality["detected_image_count"] == 5
    assert quality["candidate_image_count"] == 4
    assert quality["detection_failed_image_count"] == 1
    assert quality["excluded_by_limit_image_count"] == 1
    assert quality["rejected_image_count"] == 1
    assert quality["observation_count"] == 113
    assert quality["detected_view_count"] == 5
    assert quality["candidate_view_count"] == 4
    assert quality["rejected_view_count"] == 1
    assert quality["views"][1] == {
      "image_index": 4,
      "board_index": 0,
      "observation_count": 38,
      "RMS": 0.30
    }

    # The external-calibration loader intentionally consumes only K, dist,
    # image_size and model, ignoring the additional quality metadata.
    loaded = load_calibration(output)
    assert set(loaded.cameras) == {"Cam1"}
    assert loaded.camera_poses is None
    assert np.allclose(
      loaded.cameras["Cam1"].intrinsic, camera.intrinsic
    )
    assert np.allclose(loaded.cameras["Cam1"].dist, camera.dist)
