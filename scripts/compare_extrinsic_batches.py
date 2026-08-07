#!/usr/bin/env python3
"""Compare pairwise extrinsic stability across the four captured batches."""

import argparse
import filecmp
import json
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from multical.transform import matrix


BATCHES = [
  {
    "batch": "2026-07-29",
    "json": "20260729/calibration.json",
    "workspace": "20260729/calibration.pkl",
  },
  {
    "batch": "2026-07-31",
    "json": "20260731/extrinsic/calibration.json",
    "workspace": "20260731/extrinsic/calibration.pkl",
  },
  {
    "batch": "2026-08-03 cam2-cam3",
    "json": "20260731/20260803_112203/calibration_cam2_cam3.json",
    "workspace": "20260731/20260803_112203/calibration_cam2_cam3.pkl",
  },
  {
    "batch": "2026-08-03 cam0-cam1",
    "json": "20260731/20260803_122238/calibration_cam0_cam1.json",
    "workspace": "20260731/20260803_122238/calibration_cam0_cam1.pkl",
  },
]


def load_json(path):
  return json.loads(path.read_text(encoding="utf-8"))


def pose_matrix(value):
  result = np.eye(4, dtype=float)
  result[:3, :3] = np.asarray(value["R"], dtype=float)
  result[:3, 3] = np.asarray(value["T"], dtype=float)
  return result


def camera_pose_map(data):
  result = {}
  for name, value in data["camera_poses"].items():
    camera = name.split("_to_", 1)[0]
    result[camera] = pose_matrix(value)
  return result


def pair_transform(camera_poses, first, second):
  return camera_poses[second] @ np.linalg.inv(camera_poses[first])


def transform_error(first, second):
  errors = matrix.pose_errors(
    np.asarray(first).reshape(1, 4, 4),
    np.asarray(second).reshape(1, 4, 4),
  )
  return {
    "rotation_deg": float(errors.rotation_deg[0]),
    "translation": float(errors.translation[0]),
    "frobenius": float(errors.frobius[0]),
  }


def rms(values):
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if not values.size:
    return None
  return float(np.sqrt(np.mean(np.square(values))))


def percentile(values, q):
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if not values.size:
    return None
  return float(np.percentile(values, q))


def correlation(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  valid = np.isfinite(x) & np.isfinite(y)
  if np.count_nonzero(valid) < 4:
    return None
  x = x[valid]
  y = y[valid]
  x_rank = np.argsort(np.argsort(x)).astype(float)
  y_rank = np.argsort(np.argsort(y)).astype(float)
  if np.std(x_rank) == 0 or np.std(y_rank) == 0:
    return None
  return float(np.corrcoef(x_rank, y_rank)[0, 1])


def incidence_angle(pose):
  cosine = float(np.clip(abs(pose[2, 2]), 0.0, 1.0))
  return float(np.degrees(np.arccos(cosine)))


def projected_area_rate(points, valid, image_size):
  selected = np.asarray(points, dtype=float)[np.asarray(valid, dtype=bool)]
  if selected.shape[0] < 4:
    return None
  span = selected.max(axis=0) - selected.min(axis=0)
  width, height = image_size
  return float((span[0] * span[1]) / (width * height))


def subset_scatter(poses_first, poses_second, mask):
  mask = np.asarray(mask, dtype=bool)
  count = int(np.count_nonzero(mask))
  if count < 3:
    return {"count": count, "rotation_deg": None, "translation": None}
  try:
    transform, inliers = matrix.align_transforms_robust(
      poses_first, poses_second, valid=mask
    )
  except (ValueError, np.linalg.LinAlgError):
    return {"count": count, "rotation_deg": None, "translation": None}
  inliers = np.asarray(inliers, dtype=bool)
  errors = matrix.pose_errors(
    transform @ poses_first[inliers], poses_second[inliers]
  )
  return {
    "count": count,
    "inlier_count": int(np.count_nonzero(inliers)),
    "rotation_deg": rms(errors.rotation_deg),
    "translation": rms(errors.translation),
  }


def write_sqlite(path, tables):
  path.parent.mkdir(parents=True, exist_ok=True)
  with sqlite3.connect(path) as connection:
    for table_name, rows in tables.items():
      scalar_rows = [
        {
          key: value
          for key, value in row.items()
          if value is None or isinstance(value, (str, int, float, bool))
        }
        for row in rows
      ]
      columns = []
      for row in scalar_rows:
        for key in row:
          if key not in columns:
            columns.append(key)
      connection.execute('DROP TABLE IF EXISTS "{}"'.format(table_name))
      if not columns:
        continue
      column_types = {}
      for column in columns:
        values = [row.get(column) for row in scalar_rows]
        non_null = [value for value in values if value is not None]
        if non_null and all(isinstance(value, bool) for value in non_null):
          column_types[column] = "INTEGER"
        elif non_null and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in non_null):
          column_types[column] = "INTEGER"
        elif non_null and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in non_null):
          column_types[column] = "REAL"
        else:
          column_types[column] = "TEXT"
      definition = ", ".join(
        '"{}" {}'.format(column, column_types[column])
        for column in columns
      )
      connection.execute(
        'CREATE TABLE "{}" ({})'.format(table_name, definition)
      )
      placeholders = ", ".join("?" for _ in columns)
      quoted_columns = ", ".join('"{}"'.format(column) for column in columns)
      connection.executemany(
        'INSERT INTO "{}" ({}) VALUES ({})'.format(
          table_name, quoted_columns, placeholders
        ),
        [
          [int(row.get(column)) if isinstance(row.get(column), bool)
           else row.get(column) for column in columns]
          for row in scalar_rows
        ],
      )


def analyze_pair(batch, data, workspace, pair_name, statistics):
  first, second = statistics["cameras"]
  names = list(workspace.names.camera)
  first_index = names.index(first)
  second_index = names.index(second)
  final_transform = pair_transform(camera_pose_map(data), first, second)

  pose_table = workspace.pose_table
  point_table = workspace.point_table
  poses_first = np.asarray(pose_table.poses[first_index]).reshape(-1, 4, 4)
  poses_second = np.asarray(pose_table.poses[second_index]).reshape(-1, 4, 4)
  valid_first = np.asarray(pose_table.valid[first_index]).reshape(-1)
  valid_second = np.asarray(pose_table.valid[second_index]).reshape(-1)
  common = valid_first & valid_second
  robust_transform, robust_inliers = matrix.align_transforms_robust(
    poses_first, poses_second, valid=common
  )

  first_size = tuple(data["cameras"][first]["image_size"])
  second_size = tuple(data["cameras"][second]["image_size"])
  images = list(workspace.names.image)
  frame_rows = []
  for frame_index in np.flatnonzero(common):
    first_pose = poses_first[frame_index]
    second_pose = poses_second[frame_index]
    frame_transform = second_pose @ np.linalg.inv(first_pose)
    error = transform_error(frame_transform, final_transform)
    pair_center_error = transform_error(frame_transform, robust_transform)
    first_area = projected_area_rate(
      point_table.points[first_index, frame_index, 0],
      point_table.valid[first_index, frame_index, 0],
      first_size,
    )
    second_area = projected_area_rate(
      point_table.points[second_index, frame_index, 0],
      point_table.valid[second_index, frame_index, 0],
      second_size,
    )
    area = math.sqrt(first_area * second_area)
    frame_rows.append({
      "batch": batch,
      "pair": pair_name,
      "frame_index": int(frame_index),
      "image": images[frame_index],
      "rotation_residual_deg": error["rotation_deg"],
      "translation_residual": error["translation"],
      "frobenius_residual": error["frobenius"],
      "rotation_pair_center_residual_deg": pair_center_error[
        "rotation_deg"
      ],
      "translation_pair_center_residual": pair_center_error[
        "translation"
      ],
      "projected_area_rate": area,
      "projected_area_rate_first": first_area,
      "projected_area_rate_second": second_area,
      "distance_first": float(np.linalg.norm(first_pose[:3, 3])),
      "distance_second": float(np.linalg.norm(second_pose[:3, 3])),
      "incidence_deg_first": incidence_angle(first_pose),
      "incidence_deg_second": incidence_angle(second_pose),
      "pnp_reprojection_first": float(
        np.asarray(pose_table.reprojection_error[first_index]).reshape(-1)[frame_index]
      ),
      "pnp_reprojection_second": float(
        np.asarray(pose_table.reprojection_error[second_index]).reshape(-1)[frame_index]
      ),
    })

  area_values = [row["projected_area_rate"] for row in frame_rows]
  rotation_values = [
    row["rotation_pair_center_residual_deg"] for row in frame_rows
  ]
  translation_values = [
    row["translation_pair_center_residual"] for row in frame_rows
  ]
  distance_values = [
    math.sqrt(row["distance_first"] * row["distance_second"])
    for row in frame_rows
  ]
  incidence_values = [
    min(row["incidence_deg_first"], row["incidence_deg_second"])
    for row in frame_rows
  ]
  pnp_values = [
    max(row["pnp_reprojection_first"], row["pnp_reprojection_second"])
    for row in frame_rows
  ]
  area_median = percentile(area_values, 50)
  area_p75 = percentile(area_values, 75)
  large_area_mask = [value >= area_median for value in area_values]
  small_area_mask = [value < area_median for value in area_values]
  top_area_mask = [value >= area_p75 for value in area_values]
  common_indices = np.flatnonzero(common)
  large_area_full = np.zeros_like(common, dtype=bool)
  small_area_full = np.zeros_like(common, dtype=bool)
  top_area_full = np.zeros_like(common, dtype=bool)
  large_area_full[common_indices[np.asarray(large_area_mask, dtype=bool)]] = True
  small_area_full[common_indices[np.asarray(small_area_mask, dtype=bool)]] = True
  top_area_full[common_indices[np.asarray(top_area_mask, dtype=bool)]] = True

  large_board_half = subset_scatter(
    poses_first, poses_second, large_area_full
  )
  small_board_half = subset_scatter(
    poses_first, poses_second, small_area_full
  )
  largest_board_quartile = subset_scatter(
    poses_first, poses_second, top_area_full
  )

  summary = {
    "batch": batch,
    "pair": pair_name,
    "common_frames": statistics["common_frame_count"],
    "pose_inliers": statistics["pose_inlier_count"],
    "pair_reprojection_RMS_px": statistics["pair_reprojection_RMS_px"],
    "rotation_scatter_deg": statistics["rotation_scatter_deg"],
    "translation_scatter": statistics["translation_scatter"],
    "final_rotation_residual_rms_deg": statistics[
      "final_rotation_residual_rms_deg"
    ],
    "final_translation_residual_rms": statistics[
      "final_translation_residual_rms"
    ],
    "final_edge_rotation_residual_deg": statistics[
      "final_edge_rotation_residual_deg"
    ],
    "final_edge_translation_residual": statistics[
      "final_edge_translation_residual"
    ],
    "projected_area_rate_median": area_median,
    "projected_area_rate_p75": area_p75,
    "distance_median": percentile(distance_values, 50),
    "incidence_deg_median": percentile(incidence_values, 50),
    "incidence_deg_range": (
      percentile(incidence_values, 100) - percentile(incidence_values, 0)
      if incidence_values else None
    ),
    "corr_area_rotation": correlation(area_values, rotation_values),
    "corr_area_translation": correlation(area_values, translation_values),
    "corr_distance_rotation": correlation(distance_values, rotation_values),
    "corr_distance_translation": correlation(
      distance_values, translation_values
    ),
    "corr_incidence_rotation": correlation(
      incidence_values, rotation_values
    ),
    "corr_incidence_translation": correlation(
      incidence_values, translation_values
    ),
    "corr_pnp_rotation": correlation(pnp_values, rotation_values),
    "corr_pnp_translation": correlation(pnp_values, translation_values),
    "large_board_half": large_board_half,
    "small_board_half": small_board_half,
    "largest_board_quartile": largest_board_quartile,
    "large_half_rotation_deg": large_board_half["rotation_deg"],
    "large_half_translation": large_board_half["translation"],
    "small_half_rotation_deg": small_board_half["rotation_deg"],
    "small_half_translation": small_board_half["translation"],
    "largest_quartile_rotation_deg": largest_board_quartile[
      "rotation_deg"
    ],
    "largest_quartile_translation": largest_board_quartile[
      "translation"
    ],
    "final_transform": final_transform.tolist(),
    "robust_pair_transform": robust_transform.tolist(),
    "robust_pose_inlier_count": int(np.count_nonzero(robust_inliers)),
  }
  return summary, frame_rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output",
    default="outputs/four_batch_extrinsic_comparison.json",
  )
  parser.add_argument(
    "--sqlite-output",
    default="outputs/four_batch_extrinsic_comparison.sqlite",
  )
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]

  intrinsic_40_path = root / "20260729/intrinsic.json"
  intrinsic_30_path = root / "20260731/intrinsic/intrinsic.json"
  intrinsic_40 = load_json(intrinsic_40_path)
  intrinsic_30 = load_json(intrinsic_30_path)
  intrinsic_sensitivity = []
  for camera in intrinsic_40["cameras"]:
    first = intrinsic_40["cameras"][camera]
    second = intrinsic_30["cameras"][camera]
    fx_first = float(first["K"][0][0])
    fy_first = float(first["K"][1][1])
    fx_second = float(second["K"][0][0])
    fy_second = float(second["K"][1][1])
    intrinsic_sensitivity.append({
      "camera": camera,
      "views_40": int(first["quality"]["view_count"]),
      "views_30": int(second["quality"]["view_count"]),
      "fx_40": fx_first,
      "fx_30": fx_second,
      "fx_delta_px": fx_second - fx_first,
      "fx_delta_rate": (fx_second - fx_first) / fx_first,
      "fy_40": fy_first,
      "fy_30": fy_second,
      "fy_delta_px": fy_second - fy_first,
      "fy_delta_rate": (fy_second - fy_first) / fy_first,
      "RMS_40": float(first["quality"]["RMS"]),
      "RMS_30": float(second["quality"]["RMS"]),
    })

  raw_images_identical = True
  for camera in intrinsic_40["cameras"]:
    left = root / "20260729" / camera
    right = root / "20260731/intrinsic" / camera
    comparison = filecmp.dircmp(left, right)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
      raw_images_identical = False
      break
    if any(
      not filecmp.cmp(left / name, right / name, shallow=False)
      for name in comparison.common_files
    ):
      raw_images_identical = False
      break

  pair_summaries = []
  frame_rows = []
  batch_summaries = []
  graph_closure = []
  transforms = {}
  robust_transforms = {}
  sources = []
  for batch_config in BATCHES:
    json_path = root / batch_config["json"]
    workspace_path = root / batch_config["workspace"]
    data = load_json(json_path)
    with workspace_path.open("rb") as file:
      workspace = pickle.load(file)
    sources.extend([str(json_path), str(workspace_path)])
    quality = data["quality"]
    batch_summaries.append({
      "batch": batch_config["batch"],
      "camera_count": len(data["cameras"]),
      "reprojection_RMS_px": quality["RMS"],
      "all_points_RMS_px": quality["RMS_all"],
      "inlier_observations": quality["inlier_observation_count"],
      "observations": quality["observation_count"],
      "inlier_rate": (
        quality["inlier_observation_count"] / quality["observation_count"]
      ),
    })
    consistency = data["extrinsic_quality"]["graph"].get(
      "redundant_edge_consistency", {}
    )
    for pair_name, values in consistency.items():
      graph_closure.append({
        "batch": batch_config["batch"],
        "pair": pair_name,
        "rotation_closure_error_deg": values[
          "rotation_closure_error_deg"
        ],
        "translation_closure_error": values[
          "translation_closure_error"
        ],
        "frobenius_closure_error": values[
          "frobenius_closure_error"
        ],
      })
    for pair_name, statistics in data["extrinsic_quality"]["pairs"].items():
      if statistics["common_frame_count"] <= 0:
        continue
      summary, rows = analyze_pair(
        batch_config["batch"], data, workspace, pair_name, statistics
      )
      pair_summaries.append(summary)
      frame_rows.extend(rows)
      transforms.setdefault(pair_name, {})[batch_config["batch"]] = np.asarray(
        summary["final_transform"], dtype=float
      )
      robust_transforms.setdefault(pair_name, {})[
        batch_config["batch"]
      ] = np.asarray(summary["robust_pair_transform"], dtype=float)

  cross_batch = []
  for pair_name, pair_transforms in transforms.items():
    batches = list(pair_transforms)
    for first_index in range(len(batches)):
      for second_index in range(first_index + 1, len(batches)):
        first_batch = batches[first_index]
        second_batch = batches[second_index]
        error = transform_error(
          pair_transforms[first_batch], pair_transforms[second_batch]
        )
        cross_batch.append({
          "pair": pair_name,
          "batch_a": first_batch,
          "batch_b": second_batch,
          **error,
        })

  cross_batch_robust = []
  for pair_name, pair_transforms in robust_transforms.items():
    batches = list(pair_transforms)
    for first_index in range(len(batches)):
      for second_index in range(first_index + 1, len(batches)):
        first_batch = batches[first_index]
        second_batch = batches[second_index]
        error = transform_error(
          pair_transforms[first_batch], pair_transforms[second_batch]
        )
        cross_batch_robust.append({
          "pair": pair_name,
          "batch_a": first_batch,
          "batch_b": second_batch,
          **error,
        })

  output = {
    "sources": sources,
    "intrinsic_sources": [str(intrinsic_40_path), str(intrinsic_30_path)],
    "intrinsic_raw_images_identical": raw_images_identical,
    "intrinsic_sensitivity": intrinsic_sensitivity,
    "batch_summaries": batch_summaries,
    "graph_closure": graph_closure,
    "pair_summaries": pair_summaries,
    "cross_batch_final_transform_differences": cross_batch,
    "cross_batch_robust_pair_differences": cross_batch_robust,
    "frame_rows": frame_rows,
  }
  destination = root / args.output
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  sqlite_destination = root / args.sqlite_output
  write_sqlite(sqlite_destination, {
    "batch_summaries": batch_summaries,
    "graph_closure": graph_closure,
    "intrinsic_sensitivity": intrinsic_sensitivity,
    "pair_summaries": pair_summaries,
    "cross_batch_final": cross_batch,
    "cross_batch_robust": cross_batch_robust,
    "frame_metrics": frame_rows,
  })
  print(destination)
  print(sqlite_destination)


if __name__ == "__main__":
  main()
