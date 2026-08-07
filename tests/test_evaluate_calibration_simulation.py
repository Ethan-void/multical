import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.evaluate_calibration_simulation import evaluate_to_file


def test_evaluation_is_saved_beside_ground_truth_by_default():
  identity = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
  intrinsic = [
    [900.0, 0.0, 640.0],
    [0.0, 895.0, 360.0],
    [0.0, 0.0, 1.0]
  ]
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    calibration_path = root / "calibration.json"
    truth_path = root / "ground_truth.json"
    calibration_path.write_text(json.dumps({
      "cameras": {
        "C1": {"K": intrinsic, "dist": [[0.0] * 5]}
      },
      "camera_poses": {
        "C1": {"R": identity, "T": [0.0, 0.0, 0.0]}
      }
    }), encoding="utf-8")
    truth_path.write_text(json.dumps({
      "cameras": {
        "C1": {"K": intrinsic}
      },
      "camera_poses_relative_to_C1": {
        "C1": {"R": identity, "T": [0.0, 0.0, 0.0]}
      }
    }), encoding="utf-8")

    result, output_path = evaluate_to_file(
      calibration_path, truth_path
    )

    assert output_path == (root / "evaluation.json").resolve()
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result["summary"]["mean_abs_focal_error_px"] == 0.0


def test_evaluation_supports_custom_output_path():
  with TemporaryDirectory() as temporary:
    root = Path(temporary)
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    intrinsic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    calibration_path = root / "calibration.json"
    truth_path = root / "ground_truth.json"
    calibration_path.write_text(json.dumps({
      "cameras": {"C1": {"K": intrinsic, "dist": [[0.0] * 5]}},
      "camera_poses": {
        "C1": {"R": identity, "T": [0.0, 0.0, 0.0]}
      }
    }), encoding="utf-8")
    truth_path.write_text(json.dumps({
      "cameras": {"C1": {"K": intrinsic}},
      "camera_poses_relative_to_C1": {
        "C1": {"R": identity, "T": [0.0, 0.0, 0.0]}
      }
    }), encoding="utf-8")
    custom_output = root / "reports" / "custom.json"

    _, output_path = evaluate_to_file(
      calibration_path, truth_path, custom_output
    )

    assert output_path == custom_output.resolve()
    assert custom_output.is_file()
