#!/usr/bin/env python3
"""Export intrinsic and/or extrinsic calibration summaries to Excel.

Examples:
  uv run python scripts/analyze_calibration.py \
    --intrinsic 20260729/intrinsic.json

  uv run python scripts/analyze_calibration.py \
    --extrinsic 20260729/calibration.json

  uv run python scripts/analyze_calibration.py \
    --intrinsic 20260729/intrinsic.json \
    --extrinsic 20260729/calibration.json \
    --output 20260729/calibration_analysis.xlsx
"""

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _load_json(filename, description):
  path = Path(filename).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError("{} not found: {}".format(description, path))
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as error:
    raise ValueError(
      "{} is not valid JSON: {}".format(description, path)
    ) from error
  if not isinstance(data, dict):
    raise ValueError("{} must contain a JSON object".format(description))
  return path, data


def _finite_number(value):
  if value is None or isinstance(value, bool):
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _integer(value):
  number = _finite_number(value)
  return int(number) if number is not None else None


def _image_name(image_sets, camera, image_index):
  rgb = image_sets.get("rgb", []) if isinstance(image_sets, dict) else []
  if image_index is None or not 0 <= image_index < len(rgb):
    return None
  frame = rgb[image_index]
  if not isinstance(frame, dict):
    return None
  return frame.get(camera)


def _struct_value(value, key, default=None):
  if isinstance(value, dict):
    return value.get(key, default)
  return getattr(value, key, default)


def _load_pickle(filename, description):
  path = Path(filename).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError("{} not found: {}".format(description, path))
  os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "multical-matplotlib-cache")
  )
  with path.open("rb") as file:
    return path, pickle.load(file)


def _intrinsic_detection_path(source, override=None):
  if override:
    return Path(override).expanduser().resolve()
  candidate = source.parent / ".intrinsic" / "detections.pkl"
  return candidate if candidate.is_file() else None


def _camera_name_from_filenames(filenames, fallback):
  if isinstance(filenames, (list, tuple)) and filenames:
    parts = Path(str(filenames[0])).parts
    if len(parts) >= 2:
      return parts[-2]
  return fallback


def _intrinsic_coverage(detections_file, camera_names, selected_ids):
  if detections_file is None:
    return {}, None
  source, cache = _load_pickle(
    detections_file,
    "intrinsic detection cache"
  )
  detected_points = _struct_value(cache, "detected_points")
  cache_key = _struct_value(cache, "cache_key", {})
  filenames = _struct_value(cache_key, "filenames", [])
  image_sizes = _struct_value(cache_key, "image_sizes", [])
  if not isinstance(detected_points, (list, tuple)):
    raise ValueError(
      "intrinsic detection cache has no detected_points: {}".format(source)
    )

  coverage = {}
  for camera_index, frames in enumerate(detected_points):
    fallback_name = (
      camera_names[camera_index]
      if camera_index < len(camera_names) else str(camera_index)
    )
    camera_files = (
      filenames[camera_index]
      if camera_index < len(filenames) else []
    )
    camera_name = _camera_name_from_filenames(camera_files, fallback_name)
    size = (
      image_sizes[camera_index]
      if camera_index < len(image_sizes) else None
    )
    if not isinstance(size, (list, tuple)) or len(size) < 2:
      continue
    width = _finite_number(size[0])
    height = _finite_number(size[1])
    if width in (None, 0) or height in (None, 0):
      continue

    points = []
    for frame_index in selected_ids.get(camera_name, []):
      if not 0 <= frame_index < len(frames):
        continue
      board_detections = frames[frame_index]
      if not isinstance(board_detections, (list, tuple)):
        continue
      for detection in board_detections:
        corners = np.asarray(
          _struct_value(detection, "corners", []),
          dtype=float
        )
        if corners.ndim != 2 or corners.shape[1] != 2:
          continue
        finite = np.isfinite(corners).all(axis=1)
        if finite.any():
          points.append(corners[finite])
    if not points:
      continue

    all_points = np.concatenate(points, axis=0)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    coverage[camera_name] = {
      "image_width": width,
      "image_height": height,
      "corner_count": int(all_points.shape[0]),
      "x_min_rate": float(x_min / width),
      "x_max_rate": float(x_max / width),
      "y_min_rate": float(y_min / height),
      "y_max_rate": float(y_max / height),
      "width_rate": float((x_max - x_min) / width),
      "height_rate": float((y_max - y_min) / height)
    }
  return coverage, str(source)


def analyze_intrinsic(filename, detections_file=None):
  source, data = _load_json(filename, "intrinsic calibration")
  cameras = data.get("cameras")
  if not isinstance(cameras, dict) or not cameras:
    raise ValueError(
      "intrinsic calibration has no cameras mapping: {}".format(source)
    )

  selected_ids = {}
  for camera_name, camera in cameras.items():
    quality = camera.get("quality", {})
    views = quality.get("views", []) if isinstance(quality, dict) else []
    selected_ids[camera_name] = [
      index for index in (
        _integer(view.get("image_index"))
        for view in views
        if isinstance(view, dict)
      )
      if index is not None
    ]

  detection_path = _intrinsic_detection_path(source, detections_file)
  coverage, coverage_source = _intrinsic_coverage(
    detection_path,
    list(cameras),
    selected_ids
  )

  rows = []
  for camera_name, camera in cameras.items():
    quality = camera.get("quality", {})
    if not isinstance(quality, dict):
      quality = {}
    views = quality.get("views", [])
    views = views if isinstance(views, list) else []
    valid_views = [
      view for view in views
      if isinstance(view, dict) and _finite_number(view.get("RMS")) is not None
    ]
    worst_view = max(
      valid_views,
      key=lambda view: _finite_number(view.get("RMS")),
      default=None
    )
    worst_index = (
      _integer(worst_view.get("image_index"))
      if worst_view is not None else None
    )
    row = {
      "camera": camera_name,
      "RMS": _finite_number(quality.get("RMS")),
      "mean_view_RMS": _finite_number(quality.get("mean_view_RMS")),
      "max_view_RMS": _finite_number(quality.get("max_view_RMS")),
      "view_count": _integer(
        quality.get("view_count", quality.get("image_count"))
      ),
      "input_image_count": _integer(quality.get("input_image_count")),
      "detected_image_count": _integer(
        quality.get("detected_image_count")
      ),
      "detection_failed_image_count": _integer(
        quality.get("detection_failed_image_count")
      ),
      "quality_rejected_image_count": _integer(
        quality.get("quality_rejected_image_count")
      ),
      "excluded_by_limit_image_count": _integer(
        quality.get("excluded_by_limit_image_count")
      ),
      "rejected_image_count": _integer(
        quality.get("rejected_image_count")
      ),
      "observation_count": _integer(quality.get("observation_count")),
      "worst_view": (
        _image_name(data.get("image_sets", {}), camera_name, worst_index)
        or (
          str(worst_index) if worst_index is not None else None
        )
      )
    }
    row.update(coverage.get(camera_name, {
      "image_width": None,
      "image_height": None,
      "corner_count": None,
      "x_min_rate": None,
      "x_max_rate": None,
      "y_min_rate": None,
      "y_max_rate": None,
      "width_rate": None,
      "height_rate": None
    }))
    rows.append(row)

  return {
    "source": str(source),
    "coverage_source": coverage_source,
    "camera_count": len(rows),
    "has_quality": any(row["RMS"] is not None for row in rows),
    "has_coverage": any(row["width_rate"] is not None for row in rows),
    "rows": rows
  }


def _pose_row(name, pose):
  translation = pose.get("T") if isinstance(pose, dict) else None
  if not isinstance(translation, list) or len(translation) != 3:
    translation = [None, None, None]
  translation = [_finite_number(value) for value in translation]
  baseline = (
    math.sqrt(sum(value * value for value in translation))
    if all(value is not None for value in translation) else None
  )
  if "_to_" in name:
    camera, reference = name.split("_to_", 1)
    relation = "{} → {}".format(camera, reference)
    role = "相对位姿"
  else:
    camera = name
    relation = name
    role = "参考相机"
  return {
    "camera": camera,
    "relation": relation,
    "role": role,
    "tx": translation[0],
    "ty": translation[1],
    "tz": translation[2],
    "baseline": baseline
  }


def _extrinsic_workspace_path(source, override=None):
  if override:
    return Path(override).expanduser().resolve()
  candidate = source.with_suffix(".pkl")
  return candidate if candidate.is_file() else None


def _rms(values):
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if not values.size:
    return None
  return float(np.sqrt(np.mean(np.square(values))))


def _extrinsic_outlier_distribution(workspace_file):
  if workspace_file is None:
    return [], [], None
  source, workspace = _load_pickle(
    workspace_file,
    "extrinsic calibration workspace"
  )
  calibration = getattr(workspace, "latest_calibration", None)
  if calibration is None:
    raise ValueError(
      "workspace has no latest_calibration: {}".format(source)
    )

  valid = np.asarray(calibration.valid, dtype=bool)
  inliers = np.asarray(calibration.inliers, dtype=bool)
  observed = np.asarray(calibration.point_table.points, dtype=float)
  predicted = np.asarray(calibration.reprojected.points, dtype=float)
  if valid.shape != inliers.shape or observed.shape[:-1] != valid.shape:
    raise ValueError(
      "workspace calibration arrays have incompatible shapes: {}".format(
        source
      )
    )
  errors = np.linalg.norm(predicted - observed, axis=-1)
  outliers = valid & ~inliers

  camera_names = list(getattr(workspace.names, "camera", []))
  image_names = list(getattr(workspace.names, "image", []))
  camera_rows = []
  frame_rows = []
  for camera_index in range(valid.shape[0]):
    camera_name = (
      camera_names[camera_index]
      if camera_index < len(camera_names) else str(camera_index)
    )
    camera_valid = valid[camera_index]
    camera_inliers = inliers[camera_index]
    camera_outliers = outliers[camera_index]
    observation_count = int(camera_valid.sum())
    inlier_count = int(camera_inliers.sum())
    outlier_count = int(camera_outliers.sum())

    affected_frames = []
    whole_outlier_frames = 0
    for frame_index in range(camera_valid.shape[0]):
      frame_valid = camera_valid[frame_index]
      frame_inliers = camera_inliers[frame_index]
      frame_outliers = camera_outliers[frame_index]
      frame_observations = int(frame_valid.sum())
      frame_outlier_count = int(frame_outliers.sum())
      if frame_outlier_count == 0:
        continue
      if frame_observations and frame_outlier_count == frame_observations:
        whole_outlier_frames += 1
      image_name = (
        image_names[frame_index]
        if frame_index < len(image_names) else str(frame_index)
      )
      row = {
        "camera": camera_name,
        "image": image_name,
        "observation_count": frame_observations,
        "inlier_count": int(frame_inliers.sum()),
        "outlier_count": frame_outlier_count,
        "outlier_rate": (
          frame_outlier_count / frame_observations
          if frame_observations else None
        ),
        "RMS_all": _rms(
          errors[camera_index, frame_index][frame_valid]
        ),
        "RMS_inlier": _rms(
          errors[camera_index, frame_index][frame_inliers]
        )
      }
      affected_frames.append(row)
      frame_rows.append(row)

    worst_frame = max(
      affected_frames,
      key=lambda row: (
        row["outlier_count"],
        row["outlier_rate"] or 0,
        row["RMS_all"] or 0
      ),
      default=None
    )
    camera_rows.append({
      "camera": camera_name,
      "observation_count": observation_count,
      "inlier_count": inlier_count,
      "outlier_count": outlier_count,
      "outlier_rate": (
        outlier_count / observation_count
        if observation_count else None
      ),
      "affected_frame_count": len(affected_frames),
      "whole_outlier_frame_count": whole_outlier_frames,
      "worst_frame": worst_frame["image"] if worst_frame else None,
      "worst_frame_outlier_count": (
        worst_frame["outlier_count"] if worst_frame else 0
      )
    })

  frame_rows.sort(
    key=lambda row: (
      -row["outlier_count"],
      -(row["outlier_rate"] or 0),
      row["camera"],
      row["image"]
    )
  )
  return camera_rows, frame_rows, str(source)


def _analyze_extrinsic_quality(data):
  quality = data.get("extrinsic_quality", {})
  if not isinstance(quality, dict):
    quality = {}

  camera_rows = []
  cameras = quality.get("cameras", {})
  if isinstance(cameras, dict):
    for camera_name, statistics in cameras.items():
      statistics = statistics if isinstance(statistics, dict) else {}
      camera_rows.append({
        "camera": camera_name,
        "observation_count": _integer(
          statistics.get("observation_count")
        ),
        "inlier_count": _integer(statistics.get("inlier_count")),
        "outlier_count": _integer(statistics.get("outlier_count")),
        "inlier_ratio": _finite_number(statistics.get("inlier_ratio")),
        "detected_frame_count": _integer(
          statistics.get("detected_frame_count")
        ),
        "inlier_frame_count": _integer(
          statistics.get("inlier_frame_count")
        ),
        "rejected_frame_count": _integer(
          statistics.get("rejected_frame_count")
        ),
        "reprojection_RMS_px": _finite_number(
          statistics.get("reprojection_RMS_px")
        ),
        "all_points_RMS_px": _finite_number(
          statistics.get("all_points_RMS_px")
        )
      })

  pair_rows = []
  pairs = quality.get("pairs", {})
  if isinstance(pairs, dict):
    for pair_name, statistics in pairs.items():
      statistics = statistics if isinstance(statistics, dict) else {}
      pair_cameras = statistics.get("cameras", [])
      pair_rows.append({
        "pair": pair_name,
        "camera_a": (
          pair_cameras[0]
          if isinstance(pair_cameras, list) and pair_cameras else None
        ),
        "camera_b": (
          pair_cameras[1]
          if isinstance(pair_cameras, list) and len(pair_cameras) > 1
          else None
        ),
        "selected_initialization_edge": bool(
          statistics.get("selected_initialization_edge", False)
        ),
        "common_frame_count": _integer(
          statistics.get("common_frame_count")
        ),
        "common_corner_count": _integer(
          statistics.get("common_corner_count")
        ),
        "common_inlier_corner_count": _integer(
          statistics.get("common_inlier_corner_count")
        ),
        "pair_reprojection_RMS_px": _finite_number(
          statistics.get("pair_reprojection_RMS_px")
        ),
        "common_pose_count": _integer(
          statistics.get("common_pose_count")
        ),
        "pose_inlier_count": _integer(
          statistics.get("pose_inlier_count")
        ),
        "rotation_scatter_deg": _finite_number(
          statistics.get("rotation_scatter_deg")
        ),
        "translation_scatter": _finite_number(
          statistics.get("translation_scatter")
        ),
        "frobenius_scatter": _finite_number(
          statistics.get("frobenius_scatter")
        ),
        "final_edge_rotation_residual_deg": _finite_number(
          statistics.get("final_edge_rotation_residual_deg")
        ),
        "final_edge_translation_residual": _finite_number(
          statistics.get("final_edge_translation_residual")
        ),
        "final_rotation_residual_rms_deg": _finite_number(
          statistics.get("final_rotation_residual_rms_deg")
        ),
        "final_translation_residual_rms": _finite_number(
          statistics.get("final_translation_residual_rms")
        ),
        "status": statistics.get("status")
      })

  graph = quality.get("graph", {})
  graph = graph if isinstance(graph, dict) else {}
  closure_rows = []
  consistency = graph.get("redundant_edge_consistency", {})
  if isinstance(consistency, dict):
    for pair_name, statistics in consistency.items():
      statistics = statistics if isinstance(statistics, dict) else {}
      closure_rows.append({
        "pair": pair_name,
        "rotation_closure_error_deg": _finite_number(
          statistics.get("rotation_closure_error_deg")
        ),
        "translation_closure_error": _finite_number(
          statistics.get("translation_closure_error")
        ),
        "frobenius_closure_error": _finite_number(
          statistics.get("frobenius_closure_error")
        )
      })

  warning_rows = []
  warnings = quality.get("warnings", [])
  if isinstance(warnings, list):
    for warning in warnings:
      if not isinstance(warning, dict):
        continue
      warning_rows.append({
        "code": warning.get("code"),
        "camera": warning.get("camera"),
        "pair": warning.get("pair"),
        "message": warning.get("message")
      })

  thresholds = quality.get("thresholds", {})
  thresholds = thresholds if isinstance(thresholds, dict) else {}
  return {
    "available": bool(quality),
    "diagnostic_only": bool(quality.get("diagnostic_only", False)),
    "translation_unit": quality.get("translation_unit"),
    "thresholds": {
      status: {
        "min_common_frames": _integer(values.get("min_common_frames")),
        "max_rotation_scatter_deg": _finite_number(
          values.get("max_rotation_scatter_deg")
        ),
        "max_translation_scatter": _finite_number(
          values.get("max_translation_scatter")
        )
      }
      for status, values in thresholds.items()
      if isinstance(values, dict)
    },
    "cameras": camera_rows,
    "pairs": pair_rows,
    "graph": {
      "connected": graph.get("connected"),
      "initialization_master": graph.get("initialization_master"),
      "selected_initialization_edges": (
        graph.get("selected_initialization_edges", [])
        if isinstance(graph.get("selected_initialization_edges", []), list)
        else []
      ),
      "weak_selected_edges": (
        graph.get("weak_selected_edges", [])
        if isinstance(graph.get("weak_selected_edges", []), list)
        else []
      ),
      "redundant_edge_consistency": closure_rows
    },
    "warnings": warning_rows
  }


def analyze_extrinsic(filename, workspace_file=None):
  source, data = _load_json(filename, "extrinsic calibration")
  cameras = data.get("cameras")
  poses = data.get("camera_poses")
  if not isinstance(cameras, dict) or not cameras:
    raise ValueError(
      "extrinsic calibration has no cameras mapping: {}".format(source)
    )
  if not isinstance(poses, dict) or not poses:
    raise ValueError(
      "extrinsic calibration has no camera_poses mapping: {}".format(source)
    )

  quality = data.get("quality", {})
  quality = quality if isinstance(quality, dict) else {}
  observations = _integer(quality.get("observation_count"))
  inliers = _integer(quality.get("inlier_observation_count"))
  outliers = (
    observations - inliers
    if observations is not None and inliers is not None else None
  )
  inlier_rate = (
    inliers / observations
    if observations not in (None, 0) and inliers is not None else None
  )
  pose_rows = [_pose_row(name, pose) for name, pose in poses.items()]
  workspace_path = _extrinsic_workspace_path(source, workspace_file)
  camera_distribution, frame_distribution, workspace_source = (
    _extrinsic_outlier_distribution(workspace_path)
  )
  extrinsic_quality = _analyze_extrinsic_quality(data)

  return {
    "source": str(source),
    "workspace_source": workspace_source,
    "camera_count": len(cameras),
    "has_quality": _finite_number(quality.get("RMS")) is not None,
    "has_outlier_distribution": bool(camera_distribution),
    "summary": {
      "RMS": _finite_number(quality.get("RMS")),
      "RMS_all": _finite_number(quality.get("RMS_all")),
      "inlier_observation_count": inliers,
      "observation_count": observations,
      "outlier_observation_count": outliers,
      "inlier_rate": inlier_rate,
      "outlier_filter_applied": bool(
        quality.get("outlier_filter_applied", False)
      )
    },
    "poses": pose_rows,
    "outlier_distribution": camera_distribution,
    "outlier_frames": frame_distribution,
    "extrinsic_quality": extrinsic_quality
  }


def _export_xlsx(report, destination, preview_dir=None):
  exporter = (
    Path(__file__).resolve().parent /
    "export_calibration_analysis_xlsx.mjs"
  )
  if not exporter.is_file():
    raise RuntimeError("Excel exporter not found: {}".format(exporter))
  node = os.environ.get("MULTICAL_NODE") or shutil.which("node")
  if not node:
    raise RuntimeError("Node.js is required to generate the Excel report")

  destination.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="multical-report-") as temporary:
    report_path = Path(temporary) / "calibration_analysis.json"
    report_path.write_text(
      json.dumps(report, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8"
    )
    command = [node, str(exporter), str(report_path), str(destination)]
    if preview_dir is not None:
      command.append(str(preview_dir))
    try:
      subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
      )
    except subprocess.CalledProcessError as error:
      details = (error.stderr or error.stdout or "").strip()
      raise RuntimeError(
        "failed to generate Excel report: {}".format(details)
      ) from error
  # artifact-tool may emit a workbook-inspection sidecar beside the report.
  # It is useful for automated QA but is not a user-facing pipeline output.
  inspection_sidecar = Path(str(destination) + ".inspect.ndjson")
  try:
    inspection_sidecar.unlink()
  except FileNotFoundError:
    pass


def build_report(
    intrinsic_file=None,
    extrinsic_file=None,
    intrinsic_detections=None,
    workspace_file=None):
  if intrinsic_file is None and extrinsic_file is None:
    raise ValueError("provide --intrinsic and/or --extrinsic")
  return {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "intrinsic": (
      analyze_intrinsic(intrinsic_file, intrinsic_detections)
      if intrinsic_file is not None else None
    ),
    "extrinsic": (
      analyze_extrinsic(extrinsic_file, workspace_file)
      if extrinsic_file is not None else None
    )
  }


def default_output(intrinsic_file=None, extrinsic_file=None):
  source = intrinsic_file or extrinsic_file
  return Path(source).expanduser().resolve().parent / "calibration_analysis.xlsx"


def main():
  parser = argparse.ArgumentParser(
    description="Export intrinsic/extrinsic calibration metrics to Excel"
  )
  parser.add_argument(
    "--intrinsic",
    help="intrinsic.json to analyze"
  )
  parser.add_argument(
    "--extrinsic",
    help="calibration.json to analyze"
  )
  parser.add_argument(
    "--intrinsic_detections",
    "--intrinsic-detections",
    dest="intrinsic_detections",
    help="detections.pkl used for intrinsic coverage (auto-detected by default)"
  )
  parser.add_argument(
    "--workspace",
    help="calibration.pkl used for outlier distribution (auto-detected by default)"
  )
  parser.add_argument(
    "--output",
    help="output .xlsx path (default: next to the input files)"
  )
  parser.add_argument(
    "--preview_dir",
    help=argparse.SUPPRESS
  )
  args = parser.parse_args()
  if not args.intrinsic and not args.extrinsic:
    parser.error("at least one of --intrinsic or --extrinsic is required")

  destination = (
    Path(args.output).expanduser().resolve()
    if args.output else default_output(args.intrinsic, args.extrinsic)
  )
  if destination.suffix.lower() != ".xlsx":
    parser.error("--output must end with .xlsx")

  report = build_report(
    args.intrinsic,
    args.extrinsic,
    args.intrinsic_detections,
    args.workspace
  )
  _export_xlsx(
    report,
    destination,
    Path(args.preview_dir).expanduser().resolve()
    if args.preview_dir else None
  )
  print("Saved calibration analysis Excel to {}".format(destination))


if __name__ == "__main__":
  main()
