"""Compare a Multical calibration result with synthetic ground truth."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rotation_error_degrees(estimated, truth):
  delta = estimated @ truth.T
  cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.rad2deg(np.arccos(cosine)))


def evaluate(calibration_file, truth_file):
  calibration_path = Path(calibration_file)
  truth_path = Path(truth_file)
  calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
  truth = json.loads(truth_path.read_text(encoding="utf-8"))

  camera_results = {}
  for camera_name, camera_truth in truth["cameras"].items():
    estimated_camera = calibration["cameras"][camera_name]
    estimated_k = np.asarray(estimated_camera["K"])
    truth_k = np.asarray(camera_truth["K"])

    pose_key = camera_name if camera_name == "C1" else "{}_to_C1".format(camera_name)
    estimated_pose = calibration["camera_poses"][pose_key]
    truth_pose = truth["camera_poses_relative_to_C1"][camera_name]
    estimated_rotation = np.asarray(estimated_pose["R"])
    truth_rotation = np.asarray(truth_pose["R"])
    estimated_translation = np.asarray(estimated_pose["T"])
    truth_translation = np.asarray(truth_pose["T"])

    camera_results[camera_name] = {
      "fx_error_px": float(estimated_k[0, 0] - truth_k[0, 0]),
      "fy_error_px": float(estimated_k[1, 1] - truth_k[1, 1]),
      "cx_error_px": float(estimated_k[0, 2] - truth_k[0, 2]),
      "cy_error_px": float(estimated_k[1, 2] - truth_k[1, 2]),
      "translation_error_m": float(np.linalg.norm(
        estimated_translation - truth_translation
      )),
      "rotation_error_deg": rotation_error_degrees(
        estimated_rotation, truth_rotation
      ),
      "estimated_distortion": estimated_camera["dist"]
    }

  summary = {
    "mean_abs_focal_error_px": float(np.mean([
      (abs(result["fx_error_px"]) + abs(result["fy_error_px"])) / 2
      for result in camera_results.values()
    ])),
    "mean_translation_error_m": float(np.mean([
      result["translation_error_m"] for result in camera_results.values()
    ])),
    "max_translation_error_m": float(np.max([
      result["translation_error_m"] for result in camera_results.values()
    ])),
    "mean_rotation_error_deg": float(np.mean([
      result["rotation_error_deg"] for result in camera_results.values()
    ])),
    "max_rotation_error_deg": float(np.max([
      result["rotation_error_deg"] for result in camera_results.values()
    ]))
  }
  return {"summary": summary, "cameras": camera_results}


def evaluate_to_file(calibration_file, truth_file, output_file=None):
  """Evaluate a synthetic calibration and save its JSON report."""
  result = evaluate(calibration_file, truth_file)
  output_path = (
    Path(output_file).resolve()
    if output_file is not None
    else Path(truth_file).resolve().parent / "evaluation.json"
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
  )
  return result, output_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--calibration",
    default="calibration_simulation/calibration.json"
  )
  parser.add_argument(
    "--ground-truth",
    default="calibration_simulation/ground_truth.json"
  )
  parser.add_argument(
    "--output",
    help=(
      "output JSON path; defaults to evaluation.json beside "
      "ground_truth.json"
    )
  )
  arguments = parser.parse_args()
  evaluation, saved_path = evaluate_to_file(
    arguments.calibration,
    arguments.ground_truth,
    arguments.output
  )
  print(json.dumps(evaluation, indent=2))
  print("Saved evaluation to {}".format(saved_path), file=sys.stderr)
