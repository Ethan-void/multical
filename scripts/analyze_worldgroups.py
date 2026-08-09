#!/usr/bin/env python3
"""Analyze relative-pose quality for independently world-anchored groups."""

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from multical.io.calibration_utils import transform_from_json


def _load_json(filename, label):
  path = Path(filename).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError("{} not found: {}".format(label, path))
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as error:
    raise ValueError("{} is not valid JSON: {}".format(label, path)) from error
  if not isinstance(data, dict):
    raise ValueError("{} must contain a JSON object".format(label))
  return path, data


def _finite(value):
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _statistics(values):
  values = np.asarray([
    value for value in values if _finite(value) is not None
  ], dtype=np.float64)
  if not values.size:
    return {
      "count": 0, "mean": None, "RMS": None,
      "median": None, "p95": None, "max": None
    }
  return {
    "count": int(values.size),
    "mean": float(np.mean(values)),
    "RMS": float(np.sqrt(np.mean(values ** 2))),
    "median": float(np.median(values)),
    "p95": float(np.percentile(values, 95)),
    "max": float(np.max(values))
  }


def _rotation_angle_deg(rotation):
  vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
  return float(np.degrees(np.linalg.norm(vector)))


def _euler_zyx_deg(rotation):
  rotation = np.asarray(rotation, dtype=np.float64)
  singular = math.hypot(rotation[0, 0], rotation[1, 0]) < 1e-9
  if not singular:
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.atan2(-rotation[2, 0], math.hypot(
      rotation[0, 0], rotation[1, 0]
    ))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
  else:
    roll = math.atan2(-rotation[1, 2], rotation[1, 1])
    pitch = math.atan2(-rotation[2, 0], math.hypot(
      rotation[0, 0], rotation[1, 0]
    ))
    yaw = 0.0
  return [float(np.degrees(value)) for value in (roll, pitch, yaw)]


def _world_point_key(point):
  values = np.asarray(point, dtype=np.float64).reshape(-1)
  if values.shape != (3,) or not np.isfinite(values).all():
    return None
  return tuple(float(round(value, 6)) for value in values)


def _group_control_errors(group):
  result = {}
  checks = (group.get("joint") or {}).get("marker_checks", [])
  for check in checks if isinstance(checks, list) else []:
    for point in check.get("points", []):
      if point.get("quality_rejected"):
        continue
      key = _world_point_key(point.get("world_point"))
      error = _finite(point.get("reprojection_error_px"))
      if key is None or error is None:
        continue
      entry = result.setdefault(key, {"all": [], "inlier": []})
      entry["all"].append(error)
      if point.get("inlier") is True:
        entry["inlier"].append(error)
  return result


def _shared_control_rows(groups):
  by_group = {
    group["name"]: _group_control_errors(group) for group in groups
  }
  point_groups = {}
  for group_name, points in by_group.items():
    for point, errors in points.items():
      point_groups.setdefault(point, {})[group_name] = errors

  rows = []
  for point, group_values in sorted(point_groups.items()):
    if len(group_values) < 2:
      continue
    metrics = {}
    means = []
    for group_name, errors in sorted(group_values.items()):
      selected = errors["inlier"] or errors["all"]
      stats = _statistics(selected)
      metrics[group_name] = {
        "observation_count": len(errors["all"]),
        "inlier_count": len(errors["inlier"]),
        "mean_reprojection_error_px": stats["mean"],
        "max_reprojection_error_px": stats["max"]
      }
      if stats["mean"] is not None:
        means.append(stats["mean"])
    rows.append({
      "world_point": list(point),
      "groups": metrics,
      "group_mean_spread_px": (
        float(max(means) - min(means)) if len(means) >= 2 else None
      ),
      "worst_group_mean_px": max(means) if means else None
    })
  return rows


def _cross_group_3d(evaluation, camera_groups):
  if evaluation is None:
    return {
      "available": False,
      "summary": _statistics([]),
      "points": [],
      "failed_reconstructions": []
    }
  points = []
  for point in evaluation.get("points", []):
    cameras = point.get("cameras_used", [])
    group_names = sorted({
      camera_groups[camera] for camera in cameras if camera in camera_groups
    })
    if len(group_names) < 2:
      continue
    points.append({
      "frame": point.get("frame"),
      "groups": group_names,
      "cameras_used": cameras,
      "error_3d": _finite(point.get("error_3d")),
      "reprojection_rms_px": _finite(point.get("reprojection_rms_px")),
      "error_xyz": point.get("error_xyz")
    })
  failures = []
  for item in evaluation.get("failed_reconstructions", []):
    cameras = item.get("cameras_available", [])
    group_names = sorted({
      camera_groups[camera] for camera in cameras if camera in camera_groups
    })
    if len(group_names) >= 2:
      failures.append({
        "frame": item.get("frame"),
        "groups": group_names,
        "cameras_available": cameras,
        "reason": item.get("reason")
      })
  return {
    "available": True,
    "summary": _statistics([
      point["error_3d"] for point in points
    ]),
    "points": points,
    "failed_reconstructions": failures
  }


def analyze_worldgroups(
    world_extrinsics_file, evaluation_file=None,
    max_local_rms_px=0.5, min_world_inlier_rate=0.5,
    max_world_rms_px=2.5, min_shared_control_points=4,
    max_shared_control_mean_px=3.0, max_shared_control_spread_px=3.0,
    min_cross_group_3d_points=3, max_cross_group_mean_error=0.05,
    max_cross_group_p95_error=0.10, max_cross_group_error=0.20
):
  source, data = _load_json(
    world_extrinsics_file, "grouped world extrinsics"
  )
  if data.get("method") not in {
      "grouped_world_anchor", "grouped_constrained_bundle_adjustment"}:
    raise ValueError("world extrinsics are not a grouped world result")
  groups = data.get("groups")
  cameras = data.get("cameras")
  if not isinstance(groups, list) or len(groups) < 2:
    raise ValueError("grouped world extrinsics need at least two groups")
  if not isinstance(cameras, dict) or not cameras:
    raise ValueError("grouped world extrinsics contain no cameras")

  camera_groups = {}
  group_rows = []
  for group in groups:
    name = str(group.get("name"))
    calibration_path, calibration = _load_json(
      group.get("calibration"), "group {} calibration".format(name)
    )
    quality = calibration.get("quality") or {}
    joint = group.get("joint") or {}
    observations = int(joint.get("observation_count", 0) or 0)
    inliers = int(joint.get("inlier_count", 0) or 0)
    local_consistency = group.get("local_consistency") or {}
    for camera in group.get("cameras", []):
      if camera in camera_groups:
        raise ValueError("camera {} occurs in multiple groups".format(camera))
      camera_groups[camera] = name
    group_rows.append({
      "name": name,
      "cameras": list(group.get("cameras", [])),
      "calibration": str(calibration_path),
      "world_extrinsics": group.get("world_extrinsics"),
      "local_extrinsic_rms_px": _finite(quality.get("RMS")),
      "local_extrinsic_all_points_rms_px": _finite(quality.get("RMS_all")),
      "local_extrinsic_observation_count": quality.get("observation_count"),
      "local_extrinsic_inlier_count": quality.get(
        "inlier_observation_count"
      ),
      "world_observation_count": observations,
      "world_inlier_count": inliers,
      "world_inlier_rate": inliers / observations if observations else None,
      "world_reprojection_rms_px": _finite(
        joint.get("reprojection_rms_px")
      ),
      "world_all_points_rms_px": _finite(joint.get("all_points_rms_px")),
      "world_all_points_max_px": _finite(joint.get("all_points_max_px")),
      "local_consistency_rotation_deg": _finite(
        local_consistency.get("max_rotation_error_deg")
      ),
      "local_consistency_translation": _finite(
        local_consistency.get("max_translation_error")
      ),
      "optimization_success": bool(joint.get("optimization_success", False))
    })

  unknown = set(cameras) - set(camera_groups)
  if unknown:
    raise ValueError("cameras missing group assignment: {}".format(
      ", ".join(sorted(unknown))
    ))

  camera_rows = []
  world_to_camera = {}
  for camera_name, record in cameras.items():
    transform = transform_from_json(record["world_to_camera"])
    world_to_camera[camera_name] = transform
    position = np.linalg.inv(transform)[:3, 3]
    camera_rows.append({
      "camera": camera_name,
      "group": camera_groups[camera_name],
      "position_world": position.tolist()
    })

  pair_rows = []
  cross_group_pair_count = 0
  for source_camera, destination_camera in combinations(cameras, 2):
    transform = (
      world_to_camera[destination_camera]
      @ np.linalg.inv(world_to_camera[source_camera])
    )
    same_group = (
      camera_groups[source_camera] == camera_groups[destination_camera]
    )
    if not same_group:
      cross_group_pair_count += 1
    euler = _euler_zyx_deg(transform[:3, :3])
    source_position = np.linalg.inv(world_to_camera[source_camera])[:3, 3]
    destination_position = np.linalg.inv(
      world_to_camera[destination_camera]
    )[:3, 3]
    pair_rows.append({
      "source_camera": source_camera,
      "destination_camera": destination_camera,
      "source_group": camera_groups[source_camera],
      "destination_group": camera_groups[destination_camera],
      "pair_type": "within_group" if same_group else "cross_group",
      "baseline": float(np.linalg.norm(
        destination_position - source_position
      )),
      "translation_destination_frame": transform[:3, 3].tolist(),
      "rotation_angle_deg": _rotation_angle_deg(transform[:3, :3]),
      "roll_deg": euler[0],
      "pitch_deg": euler[1],
      "yaw_deg": euler[2]
    })

  shared_controls = _shared_control_rows(groups)
  valid_shared_controls = [
    point for point in shared_controls
    if all(
      metrics.get("inlier_count", 0) > 0
      for metrics in point["groups"].values()
    )
  ]
  shared_worst_mean = max((
    point["worst_group_mean_px"] for point in shared_controls
    if point["worst_group_mean_px"] is not None
  ), default=None)
  shared_max_spread = max((
    point["group_mean_spread_px"] for point in shared_controls
    if point["group_mean_spread_px"] is not None
  ), default=None)
  evaluation_path = None
  evaluation = None
  if evaluation_file is not None:
    evaluation_path, evaluation = _load_json(
      evaluation_file, "3D evaluation"
    )
  cross_3d = _cross_group_3d(evaluation, camera_groups)

  thresholds = {
    "max_local_rms_px": float(max_local_rms_px),
    "min_world_inlier_rate": float(min_world_inlier_rate),
    "max_world_rms_px": float(max_world_rms_px),
    "min_shared_control_points": int(min_shared_control_points),
    "max_shared_control_mean_px": float(max_shared_control_mean_px),
    "max_shared_control_spread_px": float(max_shared_control_spread_px),
    "min_cross_group_3d_points": int(min_cross_group_3d_points),
    "max_cross_group_mean_error": float(max_cross_group_mean_error),
    "max_cross_group_p95_error": float(max_cross_group_p95_error),
    "max_cross_group_error": float(max_cross_group_error)
  }
  failures = []
  for group in group_rows:
    if not group["optimization_success"]:
      failures.append("{} 世界外参优化未成功".format(
        group["name"]
      ))
    if (
        group["local_extrinsic_rms_px"] is None
        or group["local_extrinsic_rms_px"] > thresholds["max_local_rms_px"]):
      failures.append("{} 局部双目外参 RMS 超过阈值".format(
        group["name"]
      ))
    if (
        group["world_inlier_rate"] is None
        or group["world_inlier_rate"] < thresholds["min_world_inlier_rate"]):
      failures.append("{} 世界控制点内点率低于阈值".format(
        group["name"]
      ))
    if (
        group["world_reprojection_rms_px"] is None
        or group["world_reprojection_rms_px"] > thresholds["max_world_rms_px"]):
      failures.append("{} 世界控制点 RMS 超过阈值".format(group["name"]))
  if len(valid_shared_controls) < thresholds["min_shared_control_points"]:
    failures.append("有效共享世界控制点数量低于阈值")
  if (
      shared_worst_mean is None
      or shared_worst_mean > thresholds["max_shared_control_mean_px"]):
    failures.append("共享控制点最差组平均重投影误差超过阈值")
  if (
      shared_max_spread is None
      or shared_max_spread > thresholds["max_shared_control_spread_px"]):
    failures.append("共享控制点两组残差差超过阈值")
  cross_summary = cross_3d["summary"]
  if not cross_3d["available"]:
    failures.append("未提供独立跨组 3D 验证")
  else:
    if cross_summary["count"] < thresholds["min_cross_group_3d_points"]:
      failures.append("独立跨组 3D 验证点数量低于阈值")
    for metric, threshold_name in (
        ("mean", "max_cross_group_mean_error"),
        ("p95", "max_cross_group_p95_error"),
        ("max", "max_cross_group_error")):
      value = cross_summary[metric]
      if value is None or value > thresholds[threshold_name]:
        failures.append("跨组 3D {} 超过阈值".format(metric))
    if cross_3d["failed_reconstructions"]:
      failures.append("跨组 3D 验证包含重建失败点")

  return {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "method": "grouped_relative_extrinsic_quality",
    "world_units": data.get("world_units", "meters"),
    "sources": {
      "world_extrinsics": str(source),
      "evaluation3d": str(evaluation_path) if evaluation_path else None
    },
    "summary": {
      "group_count": len(group_rows),
      "camera_count": len(camera_rows),
      "camera_pair_count": len(pair_rows),
      "cross_group_pair_count": cross_group_pair_count,
      "shared_control_point_count": len(shared_controls),
      "valid_shared_control_point_count": len(valid_shared_controls),
      "shared_control_worst_group_mean_px": shared_worst_mean,
      "shared_control_max_group_spread_px": shared_max_spread,
      "cross_group_3d_point_count": cross_summary["count"]
    },
    "thresholds": thresholds,
    "acceptance": {"passed": not failures, "failures": failures},
    "groups": group_rows,
    "cameras": camera_rows,
    "camera_pairs": pair_rows,
    "shared_control_points": shared_controls,
    "cross_group_3d": cross_3d,
    "limitations": [
      "跨组相对外参通过已测量的世界控制网推导，并非来自跨组共同标定板观测。",
      "除非独立 3D 验证点已经包含测量误差，否则报告不包含世界控制点测量不确定度。",
      "calibration.pkl 只对各局部组有物理意义；本报告不会伪造全局 BA workspace。"
    ]
  }


def export_xlsx(report_file, xlsx_file, preview_dir=None):
  exporter = Path(__file__).resolve().parent / "export_worldgroups_analysis_xlsx.mjs"
  node = os.environ.get("MULTICAL_NODE") or shutil.which("node")
  if not exporter.is_file():
    raise RuntimeError("Excel exporter not found: {}".format(exporter))
  if not node:
    raise RuntimeError("Node.js is required to generate the Excel report")
  command = [node, str(exporter), str(report_file), str(xlsx_file)]
  if preview_dir is not None:
    command.append(str(preview_dir))
  try:
    subprocess.run(command, check=True, capture_output=True, text=True)
  except subprocess.CalledProcessError as error:
    details = (error.stderr or error.stdout or "").strip()
    raise RuntimeError("failed to generate Excel report: {}".format(
      details
    )) from error
  # artifact-tool may emit an inspection sidecar beside the workbook. It is
  # useful during exporter development but is not a user-facing report.
  inspection_sidecar = Path(str(xlsx_file) + ".inspect.ndjson")
  try:
    inspection_sidecar.unlink()
  except FileNotFoundError:
    pass


def main():
  parser = argparse.ArgumentParser(
    description="Analyze relative extrinsic quality across world-anchored groups"
  )
  parser.add_argument("--world_extrinsics", required=True)
  parser.add_argument("--evaluation")
  parser.add_argument("--output", required=True, help="output JSON path")
  parser.add_argument("--preview_dir", help=argparse.SUPPRESS)
  parser.add_argument("--max_local_rms_px", type=float, default=0.5)
  parser.add_argument("--min_world_inlier_rate", type=float, default=0.5)
  parser.add_argument("--max_world_rms_px", type=float, default=2.5)
  parser.add_argument("--min_shared_control_points", type=int, default=4)
  parser.add_argument("--max_shared_control_mean_px", type=float, default=3.0)
  parser.add_argument("--max_shared_control_spread_px", type=float, default=3.0)
  parser.add_argument("--min_cross_group_3d_points", type=int, default=3)
  parser.add_argument("--max_cross_group_mean_error", type=float, default=0.05)
  parser.add_argument("--max_cross_group_p95_error", type=float, default=0.10)
  parser.add_argument("--max_cross_group_error", type=float, default=0.20)
  args = parser.parse_args()

  destination = Path(args.output).expanduser().resolve()
  if destination.suffix.lower() != ".json":
    parser.error("--output must end with .json")
  report = analyze_worldgroups(
    args.world_extrinsics,
    args.evaluation,
    args.max_local_rms_px,
    args.min_world_inlier_rate,
    args.max_world_rms_px,
    args.min_shared_control_points,
    args.max_shared_control_mean_px,
    args.max_shared_control_spread_px,
    args.min_cross_group_3d_points,
    args.max_cross_group_mean_error,
    args.max_cross_group_p95_error,
    args.max_cross_group_error
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
  )
  xlsx_destination = destination.with_suffix(".xlsx")
  export_xlsx(
    destination,
    xlsx_destination,
    Path(args.preview_dir).expanduser().resolve()
    if args.preview_dir else None
  )
  print(json.dumps({
    "passed": report["acceptance"]["passed"],
    "summary": report["summary"],
    "failures": report["acceptance"]["failures"]
  }, indent=2, ensure_ascii=False))
  print("Saved grouped extrinsic analysis JSON to {}".format(destination))
  print("Saved grouped extrinsic analysis Excel to {}".format(
    xlsx_destination
  ))


if __name__ == "__main__":
  main()
