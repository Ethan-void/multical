"""Evaluate reconstructed world points against measured 3D ground truth."""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from multical.app.triangulate import load_json_or_yaml
from multical.config.arguments import run_with


UNIT_TO_METERS = {
  "m": 1.0,
  "meter": 1.0,
  "meters": 1.0,
  "metre": 1.0,
  "metres": 1.0,
  "cm": 0.01,
  "centimeter": 0.01,
  "centimeters": 0.01,
  "centimetre": 0.01,
  "centimetres": 0.01,
  "mm": 0.001,
  "millimeter": 0.001,
  "millimeters": 0.001,
  "millimetre": 0.001,
  "millimetres": 0.001
}


def unit_scale(source_unit, destination_unit):
  source = str(source_unit).strip().lower()
  destination = str(destination_unit).strip().lower()
  if source == destination:
    return 1.0
  if source not in UNIT_TO_METERS or destination not in UNIT_TO_METERS:
    raise ValueError(
      "cannot convert world units {} to {}".format(
        source_unit, destination_unit
      )
    )
  return UNIT_TO_METERS[source] / UNIT_TO_METERS[destination]


def _point3(value, description):
  point = np.asarray(value, dtype=np.float64)
  if point.shape != (3,) or not np.isfinite(point).all():
    raise ValueError("{} must be a finite [X, Y, Z]".format(description))
  return point


def parse_ground_truth(data):
  """Return frame-to-point ground truth and its declared unit."""
  if isinstance(data, dict):
    units = data.get("world_units")
    coordinate_frame = data.get("coordinate_frame", "world")
    points = data.get("points", data.get("frames"))
  else:
    units = None
    coordinate_frame = "world"
    points = data
  if coordinate_frame != "world":
    raise ValueError(
      "ground truth coordinate_frame must be world"
    )

  parsed = {}
  if isinstance(points, dict):
    entries = [
      (str(identifier), value)
      for identifier, value in points.items()
    ]
  elif isinstance(points, list):
    entries = []
    for index, item in enumerate(points):
      if not isinstance(item, dict):
        raise ValueError(
          "ground truth list item {} must be a mapping".format(index)
        )
      identifier = item.get(
        "frame", item.get("id", item.get("name"))
      )
      if identifier is None:
        raise ValueError(
          "ground truth item {} is missing frame/id/name".format(index)
        )
      value = item.get("point_world")
      if value is None:
        value = item.get("point", item.get("xyz"))
      entries.append((str(identifier), value))
  else:
    raise ValueError(
      "ground truth must contain a points mapping or list"
    )

  for identifier, value in entries:
    if identifier in parsed:
      raise ValueError(
        "duplicate ground truth identifier {}".format(identifier)
      )
    parsed[identifier] = _point3(
      value, "ground truth {}".format(identifier)
    )
  if not parsed:
    raise ValueError("ground truth contains no points")
  return parsed, units


def reconstruction_frames(data):
  if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
    raise ValueError("reconstruction must contain a frames list")
  frames = {}
  for index, frame in enumerate(data["frames"]):
    if not isinstance(frame, dict) or frame.get("frame") is None:
      raise ValueError(
        "reconstruction frame {} has no frame identifier".format(index)
      )
    identifier = str(frame["frame"])
    if identifier in frames:
      raise ValueError(
        "duplicate reconstruction frame {}".format(identifier)
      )
    frames[identifier] = frame
  return frames


def _statistics(errors):
  values = np.asarray(errors, dtype=np.float64)
  return {
    "mean": float(np.mean(values)),
    "RMS": float(np.sqrt(np.mean(np.square(values)))),
    "median": float(np.median(values)),
    "p95": float(np.percentile(values, 95)),
    "max": float(np.max(values))
  }


def export_evaluation_xlsx(json_file, xlsx_file):
  """Export the machine-readable evaluation as a human-readable workbook."""
  script = (
    Path(__file__).resolve().parents[2] /
    "scripts" / "export_evaluation3d_xlsx.mjs"
  )
  if not script.is_file():
    raise RuntimeError(
      "Excel exporter not found: {}".format(script)
    )
  node = os.environ.get("MULTICAL_NODE") or shutil.which("node")
  if not node:
    raise RuntimeError(
      "Node.js is required to generate the Excel evaluation report"
    )
  try:
    subprocess.run(
      [node, str(script), str(json_file), str(xlsx_file)],
      check=True,
      capture_output=True,
      text=True
    )
  except subprocess.CalledProcessError as error:
    details = (error.stderr or error.stdout or "").strip()
    raise RuntimeError(
      "failed to generate Excel evaluation report: {}".format(details)
    ) from error
  # Keep only the user-facing workbook; discard artifact-tool's QA sidecar.
  inspection_sidecar = Path(str(xlsx_file) + ".inspect.ndjson")
  try:
    inspection_sidecar.unlink()
  except FileNotFoundError:
    pass
  return Path(xlsx_file)


def evaluate_reconstruction(
    reconstruction_file,
    ground_truth_file,
    output_file=None,
    max_mean_error=None,
    max_p95_error=None,
    max_error=None
):
  reconstruction_path = Path(reconstruction_file).expanduser().resolve()
  ground_truth_path = Path(ground_truth_file).expanduser().resolve()
  reconstruction = load_json_or_yaml(reconstruction_path)
  ground_truth_data = load_json_or_yaml(ground_truth_path)
  if (
      not isinstance(reconstruction, dict) or
      reconstruction.get("coordinate_frame") != "world"):
    raise ValueError(
      "reconstruction coordinate_frame must be world"
    )

  measured, measured_units = parse_ground_truth(ground_truth_data)
  frames = reconstruction_frames(reconstruction)
  reconstruction_units = reconstruction.get("world_units", "meters")
  measured_units = measured_units or reconstruction_units
  scale = unit_scale(measured_units, reconstruction_units)
  measured = {
    identifier: point * scale
    for identifier, point in measured.items()
  }

  point_results = []
  failed_reconstruction = []
  missing_reconstruction = []
  for identifier, measured_point in measured.items():
    frame = frames.get(identifier)
    if frame is None:
      missing_reconstruction.append(identifier)
      continue
    if frame.get("status") != "ok" or frame.get("point_world") is None:
      failed_reconstruction.append({
        "frame": identifier,
        "reason": frame.get("reason", "reconstruction status is not ok")
      })
      continue
    reconstructed_point = _point3(
      frame["point_world"],
      "reconstructed point {}".format(identifier)
    )
    delta = reconstructed_point - measured_point
    point_results.append({
      "frame": identifier,
      "measured_world": measured_point.tolist(),
      "reconstructed_world": reconstructed_point.tolist(),
      "error_xyz": delta.tolist(),
      "abs_error_xyz": np.abs(delta).tolist(),
      "error_3d": float(np.linalg.norm(delta)),
      "cameras_used": frame.get("cameras_used"),
      "reprojection_rms_px": frame.get("reprojection_rms_px")
    })

  if point_results:
    deltas = np.asarray([
      point["error_xyz"] for point in point_results
    ])
    distances = np.asarray([
      point["error_3d"] for point in point_results
    ])
    error_statistics = _statistics(distances)
    axis_statistics = {
      axis: {
        "bias": float(np.mean(deltas[:, index])),
        "MAE": float(np.mean(np.abs(deltas[:, index]))),
        "RMS": float(np.sqrt(np.mean(np.square(deltas[:, index])))),
        "max_abs": float(np.max(np.abs(deltas[:, index])))
      }
      for index, axis in enumerate(("X", "Y", "Z"))
    }
  else:
    error_statistics = None
    axis_statistics = None

  thresholds = {
    key: float(value)
    for key, value in {
      "max_mean_error": max_mean_error,
      "max_p95_error": max_p95_error,
      "max_error": max_error
    }.items()
    if value is not None
  }
  failures = []
  if missing_reconstruction:
    failures.append(
      "{} measured points have no reconstruction".format(
        len(missing_reconstruction)
      )
    )
  if failed_reconstruction:
    failures.append(
      "{} measured points failed reconstruction".format(
        len(failed_reconstruction)
      )
    )
  if error_statistics is None:
    failures.append("no measured points were reconstructed successfully")
  else:
    checks = {
      "max_mean_error": error_statistics["mean"],
      "max_p95_error": error_statistics["p95"],
      "max_error": error_statistics["max"]
    }
    for name, threshold in thresholds.items():
      if checks[name] > threshold:
        failures.append(
          "{} {:.6g} exceeds {:.6g}".format(
            name, checks[name], threshold
          )
        )

  unmeasured = sorted(set(frames) - set(measured))
  output = {
    "coordinate_frame": "world",
    "world_units": reconstruction_units,
    "sources": {
      "reconstruction": str(reconstruction_path),
      "ground_truth": str(ground_truth_path),
      "ground_truth_input_units": measured_units,
      "ground_truth_scale_to_output_units": float(scale)
    },
    "summary": {
      "ground_truth_count": len(measured),
      "matched_count": len(point_results),
      "failed_reconstruction_count": len(failed_reconstruction),
      "missing_reconstruction_count": len(missing_reconstruction),
      "unmeasured_reconstruction_count": len(unmeasured),
      "error_3d": error_statistics,
      "axis_error": axis_statistics
    },
    "acceptance": {
      "threshold_units": reconstruction_units,
      "thresholds": thresholds,
      "passed": not failures,
      "failures": failures
    },
    "points": point_results,
    "failed_reconstructions": failed_reconstruction,
    "missing_reconstructions": missing_reconstruction,
    "unmeasured_reconstructions": unmeasured
  }

  destination = (
    Path(output_file).expanduser().resolve()
    if output_file is not None else
    reconstruction_path.parent / "evaluation3d.json"
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n",
    encoding="utf-8"
  )
  xlsx_destination = destination.with_suffix(".xlsx")
  export_evaluation_xlsx(destination, xlsx_destination)
  return output, destination, xlsx_destination


@dataclass
class Evaluate3d:
  """Compare reconstructed world points with measured 3D coordinates."""

  reconstruction: str
  ground_truth: str
  output: Optional[str] = None
  max_mean_error: Optional[float] = None
  max_p95_error: Optional[float] = None
  max_error: Optional[float] = None

  def execute(self):
    result, destination, xlsx_destination = evaluate_reconstruction(
      self.reconstruction,
      self.ground_truth,
      self.output,
      self.max_mean_error,
      self.max_p95_error,
      self.max_error
    )
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["acceptance"], indent=2))
    print("Saved 3D evaluation JSON to {}".format(destination))
    print("Saved 3D evaluation Excel to {}".format(xlsx_destination))


if __name__ == "__main__":
  run_with(Evaluate3d)
