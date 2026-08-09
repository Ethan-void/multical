"""Constrained bundle adjustment for independently calibrated camera groups."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy import optimize
from scipy.sparse import lil_matrix

from multical.app.worldmulti import load_multicamera_correspondences
from multical.config.arguments import run_with
from multical.io.calibration_utils import (
  camera_pose_matrices,
  load_calibration_json,
  transform_from_json,
  transform_to_json
)


_WORLD_CONVENTION = (
  "x_camera = R_world_to_camera * x_world + T_world_to_camera"
)


def _transform_to_parameters(transform):
  rotation, _ = cv2.Rodrigues(
    np.asarray(transform[:3, :3], dtype=np.float64)
  )
  return np.concatenate([
    rotation.reshape(3),
    np.asarray(transform[:3, 3], dtype=np.float64)
  ])


def _parameters_to_transform(parameters):
  rotation, _ = cv2.Rodrigues(
    np.asarray(parameters[:3], dtype=np.float64)
  )
  transform = np.eye(4, dtype=np.float64)
  transform[:3, :3] = rotation
  transform[:3, 3] = np.asarray(parameters[3:6], dtype=np.float64)
  return transform


def _rotation_vector(transform):
  vector, _ = cv2.Rodrigues(
    np.asarray(transform[:3, :3], dtype=np.float64)
  )
  return vector.reshape(3)


def _rotation_error_deg(first, second):
  return float(np.degrees(np.linalg.norm(
    _rotation_vector(first @ np.linalg.inv(second))
  )))


def _project(camera, transform, points):
  rotation, _ = cv2.Rodrigues(transform[:3, :3])
  projected, _ = cv2.projectPoints(
    np.asarray(points, dtype=np.float64).reshape(-1, 3),
    rotation,
    transform[:3, 3],
    np.asarray(camera["K"], dtype=np.float64),
    np.asarray(camera["dist"], dtype=np.float64).reshape(-1)
  )
  return projected.reshape(-1, 2)


def _statistics(errors):
  errors = np.asarray(errors, dtype=np.float64).reshape(-1)
  if not errors.size:
    return {
      "count": 0, "mean": None, "RMS": None,
      "median": None, "p95": None, "max": None
    }
  return {
    "count": int(errors.size),
    "mean": float(np.mean(errors)),
    "RMS": float(np.sqrt(np.mean(errors ** 2))),
    "median": float(np.median(errors)),
    "p95": float(np.percentile(errors, 95)),
    "max": float(np.max(errors))
  }


def _camera_record(world_to_camera):
  camera_to_world = np.linalg.inv(world_to_camera)
  return {
    "world_to_camera": transform_to_json(world_to_camera),
    "camera_to_world": transform_to_json(camera_to_world),
    "position_world": camera_to_world[:3, 3].tolist()
  }


def _load_workspace(filename):
  path = Path(filename).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError("local calibration workspace not found: {}".format(
      path
    ))
  with path.open("rb") as source:
    workspace = pickle.load(source)
  calibration = getattr(workspace, "latest_calibration", None)
  if calibration is None:
    raise ValueError("workspace has no latest calibration: {}".format(path))
  motion = getattr(calibration, "motion", None)
  frame_poses = getattr(motion, "frame_poses", None)
  if frame_poses is None:
    raise ValueError("workspace motion model has no frame poses: {}".format(
      path
    ))
  return path, workspace, calibration


class _JointProblem:
  def __init__(
      self, intrinsic, initial_world, groups,
      charuco_weight, world_weight,
      rotation_sigma_deg, translation_sigma, prior_weights):
    self.intrinsic = intrinsic
    self.initial_world = initial_world
    self.groups = groups
    self.charuco_weight = float(charuco_weight)
    self.world_weight = float(world_weight)
    self.rotation_sigma = np.radians(float(rotation_sigma_deg))
    self.translation_sigma = float(translation_sigma)
    self.prior_weights = {
      str(name): float(weight) for name, weight in prior_weights.items()
    }
    if self.charuco_weight <= 0 or self.world_weight <= 0:
      raise ValueError("image observation weights must be positive")
    if (
        self.rotation_sigma <= 0 or self.translation_sigma <= 0
        or set(self.prior_weights) != {
          str(group["name"]) for group in groups
        }
        or any(weight <= 0 for weight in self.prior_weights.values())):
      raise ValueError("relative-pose prior settings must be positive")

    self.slices = {}
    parameters = []

    def append(key, transform):
      start = sum(value.size for value in parameters)
      value = _transform_to_parameters(transform)
      parameters.append(value)
      self.slices[key] = slice(start, start + value.size)

    for group in groups:
      append((group["name"], "world"), group["world_to_group"])
      for camera_name in group["cameras"]:
        if camera_name != group["reference"]:
          append(
            (group["name"], "camera", camera_name),
            group["local_poses"][camera_name]
          )
      for frame_key, transform in group["frame_poses"].items():
        append((group["name"], "frame", frame_key), transform)
    self.initial_parameters = np.concatenate(parameters)
    self.world_active = {
      group["name"]: np.ones(len(group["world_observations"]), dtype=bool)
      for group in groups
    }
    self._build_residual_layout()

  def _build_residual_layout(self):
    self.residual_blocks = []
    row_count = 0

    def add(kind, group, count, dependencies, payload=None):
      nonlocal row_count
      rows = slice(row_count, row_count + count * 2)
      row_count = rows.stop
      self.residual_blocks.append({
        "kind": kind,
        "group": group,
        "rows": rows,
        "dependencies": dependencies,
        "payload": payload
      })

    for group in self.groups:
      name = group["name"]
      for block in group["charuco_blocks"]:
        dependencies = [(name, "frame", block["frame_key"])]
        if block["camera"] != group["reference"]:
          dependencies.append((name, "camera", block["camera"]))
        add(
          "charuco", name, len(block["points"]), dependencies, block
        )
      for camera_name, block in group["world_blocks"].items():
        dependencies = [(name, "world")]
        if camera_name != group["reference"]:
          dependencies.append((name, "camera", camera_name))
        add(
          "world", name, len(block["points"]), dependencies, block
        )
      for camera_name in group["cameras"]:
        if camera_name == group["reference"]:
          continue
        rows = slice(row_count, row_count + 6)
        row_count = rows.stop
        self.residual_blocks.append({
          "kind": "prior",
          "group": name,
          "rows": rows,
          "dependencies": [(name, "camera", camera_name)],
          "payload": camera_name
        })
    self.residual_count = row_count
    sparsity = lil_matrix(
      (self.residual_count, self.initial_parameters.size), dtype=np.int8
    )
    for block in self.residual_blocks:
      for key in block["dependencies"]:
        parameter_slice = self.slices[key]
        sparsity[block["rows"], parameter_slice] = 1
    self.sparsity = sparsity.tocsr()

  def _transforms(self, parameters):
    return {
      key: _parameters_to_transform(parameters[value])
      for key, value in self.slices.items()
    }

  def _local_pose(self, transforms, group, camera_name):
    if camera_name == group["reference"]:
      return np.eye(4, dtype=np.float64)
    return transforms[(group["name"], "camera", camera_name)]

  def residuals(self, parameters):
    transforms = self._transforms(parameters)
    result = np.zeros(self.residual_count, dtype=np.float64)
    groups = {group["name"]: group for group in self.groups}
    for block in self.residual_blocks:
      group = groups[block["group"]]
      payload = block["payload"]
      if block["kind"] == "charuco":
        local_pose = self._local_pose(
          transforms, group, payload["camera"]
        )
        frame_pose = transforms[
          (group["name"], "frame", payload["frame_key"])
        ]
        predicted = _project(
          self.intrinsic["cameras"][payload["camera"]],
          local_pose @ frame_pose,
          payload["points"]
        )
        values = (
          predicted - payload["image_points"]
        ) * self.charuco_weight
        result[block["rows"]] = values.reshape(-1)
      elif block["kind"] == "world":
        camera_name = payload["camera"]
        local_pose = self._local_pose(transforms, group, camera_name)
        world_to_group = transforms[(group["name"], "world")]
        predicted = _project(
          self.intrinsic["cameras"][camera_name],
          local_pose @ world_to_group,
          payload["points"]
        )
        active = self.world_active[group["name"]][payload["indices"]]
        values = (
          predicted - payload["image_points"]
        ) * self.world_weight * active[:, None]
        result[block["rows"]] = values.reshape(-1)
      else:
        camera_name = payload
        local_pose = self._local_pose(transforms, group, camera_name)
        initial = group["local_poses"][camera_name]
        delta = local_pose @ np.linalg.inv(initial)
        values = np.concatenate([
          _rotation_vector(delta) / self.rotation_sigma,
          delta[:3, 3] / self.translation_sigma
        ]) * self.prior_weights[group["name"]]
        result[block["rows"]] = values
    return result

  def solution(self, parameters):
    transforms = self._transforms(parameters)
    result = {}
    for group in self.groups:
      name = group["name"]
      world_to_group = transforms[(name, "world")]
      local_poses = {}
      world_to_cameras = {}
      for camera_name in group["cameras"]:
        local = self._local_pose(transforms, group, camera_name)
        local_poses[camera_name] = local
        world_to_cameras[camera_name] = local @ world_to_group
      result[name] = {
        "world_to_group": world_to_group,
        "local_poses": local_poses,
        "world_to_cameras": world_to_cameras
      }
    return result

  def observation_errors(self, parameters):
    transforms = self._transforms(parameters)
    result = {}
    for group in self.groups:
      name = group["name"]
      charuco_errors = []
      for block in group["charuco_blocks"]:
        local = self._local_pose(transforms, group, block["camera"])
        frame = transforms[(name, "frame", block["frame_key"])]
        predicted = _project(
          self.intrinsic["cameras"][block["camera"]],
          local @ frame,
          block["points"]
        )
        charuco_errors.extend(np.linalg.norm(
          predicted - block["image_points"], axis=1
        ))
      world_errors = np.empty(len(group["world_observations"]))
      world_to_group = transforms[(name, "world")]
      for camera_name, block in group["world_blocks"].items():
        local = self._local_pose(transforms, group, camera_name)
        predicted = _project(
          self.intrinsic["cameras"][camera_name],
          local @ world_to_group,
          block["points"]
        )
        world_errors[block["indices"]] = np.linalg.norm(
          predicted - block["image_points"], axis=1
        )
      result[name] = {
        "charuco": np.asarray(charuco_errors, dtype=np.float64),
        "world": world_errors
      }
    return result


def _prepare_groups(
    initial_world, calibration_files, workspace_files,
    correspondence_files, group_names):
  initial_groups = initial_world.get("groups", [])
  if not (
      len(calibration_files) == len(workspace_files)
      == len(correspondence_files) == len(group_names)):
    raise ValueError(
      "calibrations, workspaces, correspondences and group_names must match"
    )
  initial_by_name = {
    str(group["name"]): group for group in initial_groups
  }
  groups = []
  for name, calibration_file, workspace_file, correspondence_file in zip(
      group_names, calibration_files, workspace_files, correspondence_files):
    if name not in initial_by_name:
      raise ValueError("initial world result has no group {}".format(name))
    group_record = initial_by_name[name]
    calibration_path = Path(calibration_file).expanduser().resolve()
    calibration_json = load_calibration_json(calibration_path)
    local_poses_raw = camera_pose_matrices(calibration_json)
    cameras = list(group_record["cameras"])
    if set(cameras) != set(local_poses_raw):
      raise ValueError("group {} camera sets differ".format(name))
    reference = (
      (group_record.get("local_consistency") or {}).get("reference_camera")
      or cameras[0]
    )
    reference_pose = local_poses_raw[reference]
    local_poses = {
      camera: local_poses_raw[camera] @ np.linalg.inv(reference_pose)
      for camera in cameras
    }
    world_to_reference = transform_from_json(
      initial_world["cameras"][reference]["world_to_camera"]
    )
    workspace_path, workspace, local_calibration = _load_workspace(
      workspace_file
    )
    workspace_cameras = list(workspace.names.camera)
    if workspace_cameras != list(calibration_json["cameras"]):
      raise ValueError(
        "group {} workspace camera order differs from calibration".format(
          name
        )
      )
    workspace_reference_pose = np.asarray(
      local_calibration.camera_poses.poses[
        workspace_cameras.index(reference)
      ],
      dtype=np.float64
    )

    inliers = np.asarray(local_calibration.inliers, dtype=bool)
    observed = np.asarray(
      local_calibration.point_table.points, dtype=np.float64
    )
    frame_poses = {}
    charuco_blocks = []
    for camera_index, camera_name in enumerate(workspace_cameras):
      for frame_index in range(inliers.shape[1]):
        for board_index in range(inliers.shape[2]):
          mask = inliers[camera_index, frame_index, board_index]
          if not np.any(mask):
            continue
          frame_key = (int(frame_index), int(board_index))
          if frame_key not in frame_poses:
            if not local_calibration.motion.frame_poses.valid[frame_index]:
              raise ValueError(
                "group {} frame {} has observations but no pose".format(
                  name, frame_index
                )
              )
            board_pose = local_calibration.board_poses.poses[board_index]
            frame_poses[frame_key] = (
              workspace_reference_pose
              @ local_calibration.motion.frame_poses.poses[frame_index]
              @ board_pose
            )
          board_points = np.asarray(
            local_calibration.boards[board_index].adjusted_points,
            dtype=np.float64
          )
          charuco_blocks.append({
            "camera": camera_name,
            "frame_key": frame_key,
            "points": board_points[mask],
            "image_points": observed[
              camera_index, frame_index, board_index, mask
            ]
          })

    correspondence_path = Path(correspondence_file).expanduser().resolve()
    observations, skipped, _, mode = load_multicamera_correspondences(
      correspondence_path, calibration_json["cameras"]
    )
    world_blocks = {}
    for camera_name in cameras:
      indices = np.asarray([
        index for index, observation in enumerate(observations)
        if observation["camera"] == camera_name
      ], dtype=int)
      if not indices.size:
        continue
      world_blocks[camera_name] = {
        "camera": camera_name,
        "indices": indices,
        "points": np.asarray([
          observations[index]["world_point"] for index in indices
        ], dtype=np.float64),
        "image_points": np.asarray([
          observations[index]["image_point"] for index in indices
        ], dtype=np.float64)
      }
    groups.append({
      "name": name,
      "cameras": cameras,
      "reference": reference,
      "calibration": str(calibration_path),
      "workspace": str(workspace_path),
      "correspondences": str(correspondence_path),
      "local_poses": local_poses,
      "world_to_group": world_to_reference,
      "frame_poses": frame_poses,
      "charuco_blocks": charuco_blocks,
      "world_observations": observations,
      "world_blocks": world_blocks,
      "skipped_world_observations": skipped,
      "world_input_mode": mode,
      "initial_record": group_record
    })
  return groups


def _marker_checks(group, errors, inlier_mask):
  checks = {}
  for index, observation in enumerate(group["world_observations"]):
    key = (observation.get("capture"), observation["camera"])
    check = checks.setdefault(key, {
      "capture": observation.get("capture"),
      "camera": observation["camera"],
      "points": []
    })
    check["points"].append({
      "marker_id": observation.get("marker_id"),
      "occurrence": observation.get("occurrence"),
      "world_point": np.asarray(
        observation["world_point"], dtype=np.float64
      ).tolist(),
      "detected_image_point": np.asarray(
        observation["image_point"], dtype=np.float64
      ).tolist(),
      "reprojection_error_px": float(errors[index]),
      "inlier": bool(inlier_mask[index]),
      "quality": observation.get("quality")
    })
  return list(checks.values())


def _resolve_prior_weights(
    group_names, default_weight, group_weights=None) -> Dict[str, float]:
  default_weight = float(default_weight)
  if default_weight <= 0:
    raise ValueError("relative-pose prior settings must be positive")
  if group_weights is None:
    return {str(name): default_weight for name in group_names}
  group_weights = [float(weight) for weight in group_weights]
  if len(group_weights) != len(group_names):
    raise ValueError(
      "relative_prior_weights must match group_names"
    )
  if any(weight <= 0 for weight in group_weights):
    raise ValueError("relative-pose prior settings must be positive")
  return {
    str(name): weight for name, weight in zip(group_names, group_weights)
  }


def _derive_calibration(
    intrinsic, cameras, world_output, destination, master, groups):
  master = master or next(iter(intrinsic["cameras"]))
  if master not in cameras:
    raise ValueError("master camera {} is unavailable".format(master))
  world_to_master = transform_from_json(
    cameras[master]["world_to_camera"]
  )
  camera_poses = {master: transform_to_json(np.eye(4))}
  for camera_name in intrinsic["cameras"]:
    if camera_name == master:
      continue
    world_to_camera = transform_from_json(
      cameras[camera_name]["world_to_camera"]
    )
    camera_poses[
      "{}_to_{}".format(camera_name, master)
    ] = transform_to_json(
      world_to_camera @ np.linalg.inv(world_to_master)
    )
  output = {
    "cameras": intrinsic["cameras"],
    "camera_poses": camera_poses,
    "provenance": {
      "method": "derived_from_grouped_constrained_bundle_adjustment",
      "world_extrinsics": str(world_output),
      "master": master,
      "groups": groups
    }
  }
  path = Path(destination).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
  return path


def constrained_group_bundle_adjustment(
    intrinsic_file, initial_world_file, calibration_files,
    workspace_files, correspondence_files, output_file,
    calibration_output=None, group_names=None, master=None,
    ransac_threshold=3.0, loss="soft_l1", charuco_weight=1.0,
    world_weight=1.0, relative_rotation_sigma_deg=0.1,
    relative_translation_sigma=0.02, relative_prior_weight=5.0,
    relative_prior_weights=None, max_iterations=100,
    final_recheck_iterations=0):
  intrinsic_path = Path(intrinsic_file).expanduser().resolve()
  initial_path = Path(initial_world_file).expanduser().resolve()
  intrinsic = load_calibration_json(intrinsic_path)
  initial_world = load_calibration_json(initial_path)
  if initial_world.get("convention") != _WORLD_CONVENTION:
    raise ValueError("initial world result has an unsupported convention")
  calibration_files = list(calibration_files)
  workspace_files = list(workspace_files)
  correspondence_files = list(correspondence_files)
  if group_names is None:
    group_names = [
      str(group["name"]) for group in initial_world.get("groups", [])
    ]
  else:
    group_names = list(group_names)
  groups = _prepare_groups(
    initial_world,
    calibration_files,
    workspace_files,
    correspondence_files,
    group_names
  )
  prior_weights = _resolve_prior_weights(
    group_names, relative_prior_weight, relative_prior_weights
  )
  problem = _JointProblem(
    intrinsic,
    initial_world,
    groups,
    charuco_weight,
    world_weight,
    relative_rotation_sigma_deg,
    relative_translation_sigma,
    prior_weights
  )
  initial_errors = problem.observation_errors(problem.initial_parameters)
  warmup = optimize.least_squares(
    problem.residuals,
    problem.initial_parameters,
    jac_sparsity=problem.sparsity,
    method="trf",
    x_scale="jac",
    loss=loss,
    f_scale=float(ransac_threshold),
    max_nfev=int(max_iterations)
  )
  current = warmup
  final = warmup
  recheck_rounds = max(int(final_recheck_iterations), 0)
  for _ in range(recheck_rounds):
    current_errors = problem.observation_errors(current.x)
    masks = {}
    for group in groups:
      name = group["name"]
      masks[name] = (
        current_errors[name]["world"] <= float(ransac_threshold)
      )
      if np.count_nonzero(masks[name]) < 4:
        raise RuntimeError(
          "group {} retained fewer than four world observations".format(name)
        )
    unchanged = all(np.array_equal(
      masks[group["name"]], problem.world_active[group["name"]]
    ) for group in groups)
    problem.world_active.update(masks)
    final = optimize.least_squares(
      problem.residuals,
      current.x,
      jac_sparsity=problem.sparsity,
      method="trf",
      x_scale="jac",
      loss=loss,
      f_scale=float(ransac_threshold),
      max_nfev=int(max_iterations)
    )
    current = final
    if unchanged:
      break
  final_errors = problem.observation_errors(final.x)
  solution = problem.solution(final.x)
  cameras = {}
  output_groups = []
  for group in groups:
    name = group["name"]
    group_solution = solution[name]
    local_changes = {}
    max_rotation = 0.0
    max_translation = 0.0
    for camera_name in group["cameras"]:
      optimized_local = group_solution["local_poses"][camera_name]
      initial_local = group["local_poses"][camera_name]
      rotation = _rotation_error_deg(optimized_local, initial_local)
      translation = float(np.linalg.norm(
        (optimized_local @ np.linalg.inv(initial_local))[:3, 3]
      ))
      max_rotation = max(max_rotation, rotation)
      max_translation = max(max_translation, translation)
      local_changes[camera_name] = {
        "rotation_error_deg": rotation,
        "translation_error": translation
      }
      cameras[camera_name] = _camera_record(
        group_solution["world_to_cameras"][camera_name]
      )
    world_errors = final_errors[name]["world"]
    world_inliers = world_errors <= float(ransac_threshold)
    inlier_errors = world_errors[world_inliers]
    camera_statistics = {}
    for camera_name in group["cameras"]:
      indices = group["world_blocks"].get(camera_name, {}).get(
        "indices", np.asarray([], dtype=int)
      )
      if not len(indices):
        continue
      errors = world_errors[indices]
      inliers = world_inliers[indices]
      camera_statistics[camera_name] = {
        "observation_count": int(len(indices)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "reprojection_rms_px": (
          _statistics(errors[inliers])["RMS"] if np.any(inliers) else None
        ),
        "all_points_rms_px": _statistics(errors)["RMS"],
        "all_points_max_px": _statistics(errors)["max"]
      }
    joint = {
      "input_mode": group["world_input_mode"],
      "observation_count": len(group["world_observations"]),
      "distinct_world_point_count": len({
        tuple(np.asarray(value["world_point"]).tolist())
        for value in group["world_observations"]
      }),
      "participating_cameras": sorted(camera_statistics),
      "inlier_count": int(np.count_nonzero(world_inliers)),
      "inlier_indices": np.flatnonzero(world_inliers).tolist(),
      "reprojection_rms_px": _statistics(inlier_errors)["RMS"],
      "reprojection_mean_px": _statistics(inlier_errors)["mean"],
      "reprojection_max_px": _statistics(inlier_errors)["max"],
      "all_points_rms_px": _statistics(world_errors)["RMS"],
      "all_points_max_px": _statistics(world_errors)["max"],
      "point_errors_px": world_errors.tolist(),
      "loss": loss,
      "ransac_threshold_px": float(ransac_threshold),
      "optimization_success": bool(final.success),
      "hard_outlier_rejection_applied": bool(
        int(final_recheck_iterations) > 0
      ),
      "camera_statistics": camera_statistics,
      "skipped_observations": group["skipped_world_observations"],
      "marker_checks": _marker_checks(
        group, world_errors, world_inliers
      )
    }
    initial_record = dict(group["initial_record"])
    initial_record["initial_joint"] = initial_record.get("joint")
    initial_record["joint"] = joint
    initial_record["workspace"] = group["workspace"]
    initial_record["correspondences"] = group["correspondences"]
    initial_record["local_consistency"] = {
      "reference_camera": group["reference"],
      "relative_prior_weight": prior_weights[name],
      "max_rotation_error_deg": max_rotation,
      "max_translation_error": max_translation,
      "cameras": local_changes
    }
    output_groups.append(initial_record)

  joint_summary = {
    "optimization_success": bool(final.success),
    "optimization_message": str(final.message),
    "warmup_success": bool(warmup.success),
    "loss": loss,
    "ransac_threshold_px": float(ransac_threshold),
    "charuco_weight": float(charuco_weight),
    "world_weight": float(world_weight),
    "relative_rotation_sigma_deg": float(relative_rotation_sigma_deg),
    "relative_translation_sigma": float(relative_translation_sigma),
    "relative_prior_weight": float(relative_prior_weight),
    "relative_prior_weights": prior_weights,
    "hard_outlier_rejection_applied": bool(
      int(final_recheck_iterations) > 0
    ),
    "initial_charuco": _statistics(np.concatenate([
      initial_errors[group["name"]]["charuco"] for group in groups
    ])),
    "final_charuco": _statistics(np.concatenate([
      final_errors[group["name"]]["charuco"] for group in groups
    ])),
    "initial_world": _statistics(np.concatenate([
      initial_errors[group["name"]]["world"] for group in groups
    ])),
    "final_world_all": _statistics(np.concatenate([
      final_errors[group["name"]]["world"] for group in groups
    ])),
    "final_world_inlier": _statistics(np.concatenate([
      final_errors[group["name"]]["world"][
        final_errors[group["name"]]["world"] <= float(ransac_threshold)
      ]
      for group in groups
    ]))
  }
  output = {
    "convention": _WORLD_CONVENTION,
    "method": "grouped_constrained_bundle_adjustment",
    "world_units": initial_world.get("world_units", "meters"),
    "intrinsic": str(intrinsic_path),
    "initial_world_extrinsics": str(initial_path),
    "groups": output_groups,
    "joint_bundle_adjustment": joint_summary,
    "cameras": {
      name: cameras[name] for name in intrinsic["cameras"]
    }
  }
  destination = Path(output_file).expanduser().resolve()
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
  )
  calibration_path = None
  if calibration_output is not None:
    calibration_path = _derive_calibration(
      intrinsic,
      output["cameras"],
      destination,
      calibration_output,
      master,
      group_names
    )
  return output, destination, calibration_path


@dataclass
class Worldgroupba:
  """Jointly refine grouped camera/world poses with local stereo priors."""

  intrinsic: str
  initial_world_extrinsics: str
  calibrations: List[str]
  workspaces: List[str]
  correspondences: List[str]
  output: str
  calibration_output: Optional[str] = None
  group_names: Optional[List[str]] = None
  master: Optional[str] = None
  ransac_threshold: float = 3.0
  loss: str = "soft_l1"
  charuco_weight: float = 1.0
  world_weight: float = 1.0
  relative_rotation_sigma_deg: float = 0.1
  relative_translation_sigma: float = 0.02
  relative_prior_weight: float = 5.0
  relative_prior_weights: Optional[List[float]] = None
  max_iterations: int = 100
  final_recheck_iterations: int = 0

  def execute(self):
    result, output_path, calibration_path = (
      constrained_group_bundle_adjustment(
        self.intrinsic,
        self.initial_world_extrinsics,
        self.calibrations,
        self.workspaces,
        self.correspondences,
        self.output,
        self.calibration_output,
        self.group_names,
        self.master,
        self.ransac_threshold,
        self.loss,
        self.charuco_weight,
        self.world_weight,
        self.relative_rotation_sigma_deg,
        self.relative_translation_sigma,
        self.relative_prior_weight,
        self.relative_prior_weights,
        self.max_iterations,
        self.final_recheck_iterations
      )
    )
    print(json.dumps(result["joint_bundle_adjustment"], indent=2))
    print("Saved constrained grouped world extrinsics to {}".format(
      output_path
    ))
    if calibration_path is not None:
      print("Saved constrained global calibration to {}".format(
        calibration_path
      ))


if __name__ == "__main__":
  run_with(Worldgroupba)
