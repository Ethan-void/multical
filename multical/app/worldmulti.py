"""Jointly anchor a calibrated multi-camera rig to a world frame."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from scipy import optimize

from multical.config.arguments import run_with
from multical.io.calibration_utils import (
  camera_pose_matrices,
  load_calibration_json,
  transform_from_rt,
  transform_to_json
)
from multical.app.world import (
  detect_marker_centers,
  resolve_capture_image
)


_MARKER_QUALITY_DEFAULTS = {
  "mode": "reject",
  "min_edge_px": 15.0,
  "warn_edge_px": 25.0,
  "min_side_ratio": 0.20,
  "min_area_ratio": 0.15,
  "max_view_angle_deg": 60.0,
  "warn_view_angle_deg": 45.0
}


def _load_data(filename):
  path = Path(filename).resolve()
  text = path.read_text(encoding="utf-8")
  data = (
    json.loads(text)
    if path.suffix.lower() == ".json"
    else yaml.safe_load(text)
  )
  if not isinstance(data, dict):
    raise ValueError("correspondences must be a YAML/JSON mapping")
  return path, data


def _point(value, dimensions, label):
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (dimensions,) or not np.isfinite(result).all():
    raise ValueError(
      "{} must be a finite {}-element point".format(label, dimensions)
    )
  return result


def _marker_quality_options(data, overrides=None):
  """Resolve and validate marker geometry quality thresholds."""
  options = dict(_MARKER_QUALITY_DEFAULTS)
  configured = data.get("marker_quality")
  if isinstance(configured, str):
    options["mode"] = configured
  elif configured is not None:
    if not isinstance(configured, dict):
      raise ValueError("marker_quality must be a mode or mapping")
    unknown = set(configured) - set(options)
    if unknown:
      raise ValueError(
        "unknown marker_quality options {}".format(
          ", ".join(sorted(unknown))
        )
      )
    options.update(configured)
  if overrides:
    options.update({
      key: value for key, value in overrides.items()
      if value is not None
    })

  options["mode"] = str(options["mode"]).lower()
  if options["mode"] not in {"off", "warn", "reject"}:
    raise ValueError("marker_quality mode must be off, warn, or reject")
  for key in (
      "min_edge_px", "warn_edge_px", "min_side_ratio",
      "min_area_ratio", "max_view_angle_deg",
      "warn_view_angle_deg"):
    value = options[key]
    if value is None:
      continue
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
      raise ValueError("marker_quality {} must be non-negative".format(key))
    options[key] = value
  for key in ("min_side_ratio", "min_area_ratio"):
    if options[key] is not None and options[key] > 1.0:
      raise ValueError("marker_quality {} must not exceed 1".format(key))
  for key in ("max_view_angle_deg", "warn_view_angle_deg"):
    if options[key] is not None and options[key] >= 90.0:
      raise ValueError("marker_quality {} must be below 90".format(key))
  return options


def _marker_view_angle(corners, camera):
  """Estimate angle between the square normal and camera viewing ray."""
  object_points = np.asarray([
    [-0.5, 0.5, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0]
  ], dtype=np.float64)
  try:
    success, rotation_vector, translation = cv2.solvePnP(
      object_points,
      np.asarray(corners, dtype=np.float64).reshape(4, 2),
      np.asarray(camera["K"], dtype=np.float64),
      np.asarray(camera["dist"], dtype=np.float64).reshape(-1),
      flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
  except cv2.error:
    return None
  if not success:
    return None
  rotation, _ = cv2.Rodrigues(rotation_vector)
  normal = rotation[:, 2]
  view_ray = np.asarray(translation, dtype=np.float64).reshape(3)
  norm = float(np.linalg.norm(view_ray))
  if norm <= 1e-12:
    return None
  cosine = abs(float(np.dot(normal, view_ray / norm)))
  return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def marker_geometry_quality(corners, camera, options):
  """Measure whether detected square geometry supports a stable center."""
  points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
  edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
  min_edge = float(np.min(edges))
  max_edge = float(np.max(edges))
  side_ratio = min_edge / max_edge if max_edge > 1e-12 else 0.0
  area = abs(float(cv2.contourArea(points.astype(np.float32))))
  area_ratio = area / (max_edge * max_edge) if max_edge > 1e-12 else 0.0
  view_angle = _marker_view_angle(points, camera)

  failed = []
  if (
      options["min_edge_px"] is not None
      and min_edge < options["min_edge_px"]):
    failed.append("min_edge_px")
  if (
      options["min_side_ratio"] is not None
      and side_ratio < options["min_side_ratio"]):
    failed.append("min_side_ratio")
  if (
      options["min_area_ratio"] is not None
      and area_ratio < options["min_area_ratio"]):
    failed.append("min_area_ratio")
  if options["max_view_angle_deg"] is not None:
    if view_angle is None:
      failed.append("view_angle_unavailable")
    elif view_angle > options["max_view_angle_deg"]:
      failed.append("max_view_angle_deg")

  warnings = []
  if (
      options["warn_edge_px"] is not None
      and min_edge < options["warn_edge_px"]
      and "min_edge_px" not in failed):
    warnings.append("small_marker")
  if (
      options["warn_view_angle_deg"] is not None
      and view_angle is not None
      and view_angle > options["warn_view_angle_deg"]
      and "max_view_angle_deg" not in failed):
    warnings.append("oblique_view")

  if options["mode"] == "off":
    status = "off"
  elif failed and options["mode"] == "reject":
    status = "rejected"
  elif failed or warnings:
    status = "warning"
  else:
    status = "accepted"
  return {
    "status": status,
    "min_edge_px": min_edge,
    "max_edge_px": max_edge,
    "side_ratio": float(side_ratio),
    "area_px2": area,
    "area_ratio": float(area_ratio),
    "view_angle_deg": view_angle,
    "failed_checks": failed,
    "warnings": warnings
  }


def _manual_observations(data, cameras):
  entries = data.get("observations")
  if not isinstance(entries, list) or not entries:
    raise ValueError("observations must be a non-empty list")
  observations = []
  for index, entry in enumerate(entries):
    if not isinstance(entry, dict):
      raise ValueError("observation {} must be a mapping".format(index))
    camera = str(entry.get("camera", ""))
    if camera not in cameras:
      raise ValueError(
        "observation {} has unknown camera {}".format(index, camera)
      )
    observations.append({
      "camera": camera,
      "world_point": _point(
        entry.get("world_point"), 3,
        "observation {} world_point".format(index)
      ),
      "image_point": _point(
        entry.get("image_point", entry.get("pixel")), 2,
        "observation {} image_point".format(index)
      ),
      "capture": str(entry.get("capture", index)),
      "marker_id": entry.get("marker_id")
    })
  return observations, []


def _marker_specs(capture, default_family, capture_index):
  markers = capture.get("markers")
  if not isinstance(markers, (dict, list)) or not markers:
    raise ValueError(
      "capture {} markers must be a non-empty mapping or list".format(
        capture_index
      )
    )
  entries = (
    list(markers.items()) if isinstance(markers, dict)
    else [(None, value) for value in markers]
  )
  specs = []
  for key, value in entries:
    if isinstance(markers, list):
      if not isinstance(value, dict) or "marker_id" not in value:
        raise ValueError(
          "capture {} marker list entries need marker_id".format(
            capture_index
          )
        )
      marker_id = int(value["marker_id"])
    else:
      marker_id = int(key)
    occurrence = None
    if isinstance(value, dict):
      family = str(value.get("marker_family", default_family))
      occurrence = value.get("occurrence")
      world_value = value.get("world_point")
    else:
      family = str(default_family)
      world_value = value
    if occurrence is not None:
      occurrence = {
        "top": "upper", "bottom": "lower"
      }.get(str(occurrence).lower(), str(occurrence).lower())
      if occurrence not in {"upper", "lower"}:
        raise ValueError("marker occurrence must be upper or lower")
    specs.append({
      "marker_id": marker_id,
      "family": family,
      "occurrence": occurrence,
      "world_point": _point(
        world_value, 3,
        "capture {} marker {} world_point".format(
          capture_index, marker_id
        )
      )
    })
  return specs


def _capture_image(data, capture, camera, directory):
  images = capture.get("images")
  if isinstance(images, dict):
    value = images.get(camera)
    if not value:
      return None
    path = Path(value)
    if not path.is_absolute():
      path = directory / path
    if not path.is_file():
      raise ValueError("capture image does not exist {}".format(path))
    return path

  camera_data = dict(data)
  camera_data["camera"] = camera
  try:
    return resolve_capture_image(camera_data, capture, directory)
  except ValueError as error:
    # A missing camera image means that camera did not participate in this
    # capture. Explicit bad paths and ambiguous layouts remain errors.
    if "could not find image" in str(error):
      return None
    raise


def _marker_observations(data, directory, cameras, quality_options):
  captures = data.get("captures")
  if not isinstance(captures, list) or not captures:
    raise ValueError("captures must be a non-empty list")
  requested = data.get("cameras", list(cameras))
  unknown = set(requested) - set(cameras)
  if unknown:
    raise ValueError(
      "unknown cameras {}".format(", ".join(sorted(unknown)))
    )
  default_family = data.get("marker_family", "6X6_250")
  observations = []
  skipped = []
  for capture_index, capture in enumerate(captures):
    specs = _marker_specs(capture, default_family, capture_index)
    capture_name = str(capture.get(
      "name", capture.get("frame", capture_index)
    ))
    for camera_name in requested:
      image_path = _capture_image(
        data, capture, camera_name, directory
      )
      if image_path is None:
        skipped.append({
          "capture": capture_name,
          "camera": camera_name,
          "reason": "image_not_found"
        })
        continue
      detections = {}
      for family in dict.fromkeys(spec["family"] for spec in specs):
        _, found, dictionary_name = detect_marker_centers(
          image_path, family, cameras[camera_name],
          preserve_duplicates=True
        )
        detections[family] = (found, dictionary_name)
      for spec in specs:
        found, dictionary_name = detections[spec["family"]]
        instances = sorted(
          found.get(spec["marker_id"], []),
          key=lambda item: float(item["center"][1])
        )
        occurrence = spec["occurrence"]
        if not instances or (occurrence is not None and len(instances) < 2):
          skipped.append({
            "capture": capture_name,
            "camera": camera_name,
            "marker_id": spec["marker_id"],
            "occurrence": occurrence,
            "reason": "marker_not_detected"
          })
          continue
        if occurrence == "upper":
          detection = instances[0]
        elif occurrence == "lower":
          detection = instances[-1]
        elif len(instances) == 1:
          detection = instances[0]
        else:
          raise ValueError(
            "capture {} camera {} detected duplicate marker {}; set "
            "occurrence upper/lower".format(
              capture_name, camera_name, spec["marker_id"]
              )
          )
        quality = marker_geometry_quality(
          detection["corners"], cameras[camera_name], quality_options
        )
        if quality["status"] == "rejected":
          skipped.append({
            "capture": capture_name,
            "camera": camera_name,
            "image": str(image_path.resolve()),
            "marker_id": spec["marker_id"],
            "marker_family": dictionary_name,
            "occurrence": occurrence,
            "world_point": spec["world_point"].tolist(),
            "image_point": np.asarray(
              detection["center"], dtype=np.float64
            ).tolist(),
            "corners": np.asarray(detection["corners"]).tolist(),
            "reason": "marker_quality_rejected",
            "quality": quality
          })
          continue
        observations.append({
          "camera": camera_name,
          "world_point": spec["world_point"],
          "image_point": np.asarray(
            detection["center"], dtype=np.float64
          ),
          "capture": capture_name,
          "image": str(image_path.resolve()),
          "marker_id": spec["marker_id"],
          "marker_family": dictionary_name,
          "occurrence": occurrence,
          "corners": np.asarray(detection["corners"]).tolist(),
          "quality": quality
        })
  return observations, skipped


def load_multicamera_correspondences(
    filename, cameras, marker_quality_overrides=None):
  path, data = _load_data(filename)
  if "observations" in data:
    observations, skipped = _manual_observations(data, cameras)
    mode = "manual_points"
  elif "captures" in data:
    quality_options = _marker_quality_options(
      data, marker_quality_overrides
    )
    observations, skipped = _marker_observations(
      data, path.parent, cameras, quality_options
    )
    mode = "marker_centers"
  else:
    raise ValueError(
      "correspondences need observations or captures"
    )
  if len(observations) < 4:
    raise ValueError("at least four multi-camera observations are required")
  if len({
      tuple(item["world_point"]) for item in observations
  }) < 4:
    raise ValueError("at least four distinct world points are required")
  participating = {item["camera"] for item in observations}
  if len(participating) < 2:
    raise ValueError(
      "worldmulti requires observations from at least two cameras; "
      "use the existing world command for a single camera"
    )
  return observations, skipped, data, mode


def _parameter_to_transform(parameters):
  rotation, _ = cv2.Rodrigues(
    np.asarray(parameters[:3], dtype=np.float64)
  )
  return transform_from_rt(rotation, parameters[3:6])


def _transform_to_parameter(transform):
  rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
  return np.concatenate([rotation_vector.reshape(3), transform[:3, 3]])


def _project_observation(parameters, observation, calibration, rig_poses):
  world_to_rig = _parameter_to_transform(parameters)
  world_to_camera = rig_poses[observation["camera"]] @ world_to_rig
  rotation_vector, _ = cv2.Rodrigues(world_to_camera[:3, :3])
  camera = calibration["cameras"][observation["camera"]]
  projected, _ = cv2.projectPoints(
    observation["world_point"].reshape(1, 3),
    rotation_vector,
    world_to_camera[:3, 3],
    np.asarray(camera["K"], dtype=np.float64),
    np.asarray(camera["dist"], dtype=np.float64).reshape(-1)
  )
  return projected.reshape(2)


def _residuals(parameters, observations, calibration, rig_poses):
  return np.concatenate([
    _project_observation(
      parameters, observation, calibration, rig_poses
    ) - observation["image_point"]
    for observation in observations
  ])


def _initial_candidates(observations, calibration, rig_poses):
  candidates = []
  for camera_name in calibration["cameras"]:
    camera_observations = [
      value for value in observations
      if value["camera"] == camera_name
    ]
    if len(camera_observations) < 4 or len({
        tuple(value["world_point"]) for value in camera_observations
    }) < 4:
      continue
    world = np.asarray([
      value["world_point"] for value in camera_observations
    ], dtype=np.float64)
    image = np.asarray([
      value["image_point"] for value in camera_observations
    ], dtype=np.float64)
    camera = calibration["cameras"][camera_name]
    success, rvec, translation = cv2.solvePnP(
      world, image,
      np.asarray(camera["K"], dtype=np.float64),
      np.asarray(camera["dist"], dtype=np.float64).reshape(-1),
      flags=cv2.SOLVEPNP_ITERATIVE
    )
    if success:
      rotation, _ = cv2.Rodrigues(rvec)
      world_to_camera = transform_from_rt(rotation, translation)
      world_to_rig = np.linalg.inv(
        rig_poses[camera_name]
      ) @ world_to_camera
      candidates.append((camera_name, _transform_to_parameter(world_to_rig)))
  if not candidates:
    raise ValueError(
      "initialization needs at least one camera observing four distinct "
      "world points"
    )
  return candidates


def solve_joint_world(
    calibration, observations, ransac_threshold=3.0,
    loss="soft_l1"
):
  rig_poses = camera_pose_matrices(calibration)
  candidates = _initial_candidates(
    observations, calibration, rig_poses
  )
  scored = []
  for camera_name, parameters in candidates:
    errors = np.linalg.norm(
      _residuals(
        parameters, observations, calibration, rig_poses
      ).reshape(-1, 2),
      axis=1
    )
    scored.append((
      int(np.count_nonzero(errors <= ransac_threshold)),
      -float(np.median(errors)),
      camera_name,
      parameters
    ))
  _, _, initial_camera, initial = max(
    scored, key=lambda item: (item[0], item[1])
  )
  first = optimize.least_squares(
    _residuals,
    initial,
    args=(observations, calibration, rig_poses),
    loss=loss,
    f_scale=float(ransac_threshold)
  )
  first_errors = np.linalg.norm(
    _residuals(
      first.x, observations, calibration, rig_poses
    ).reshape(-1, 2),
    axis=1
  )
  inlier_mask = first_errors <= float(ransac_threshold)
  if np.count_nonzero(inlier_mask) < 4:
    raise RuntimeError(
      "joint world fit retained fewer than four observations"
    )
  inlier_observations = [
    value for value, keep in zip(observations, inlier_mask) if keep
  ]
  final = optimize.least_squares(
    _residuals,
    first.x,
    args=(inlier_observations, calibration, rig_poses),
    loss=loss,
    f_scale=float(ransac_threshold)
  )
  errors = np.linalg.norm(
    _residuals(
      final.x, observations, calibration, rig_poses
    ).reshape(-1, 2),
    axis=1
  )
  inlier_mask = errors <= float(ransac_threshold)
  return (
    _parameter_to_transform(final.x),
    rig_poses,
    errors,
    inlier_mask,
    initial_camera,
    final
  )


def write_multicamera_check_images(
    calibration, rig_poses, world_to_rig, observations,
    errors, inlier_mask, destination, skipped_observations=()
):
  """Write detected-vs-projected marker overlays for every camera/capture."""
  grouped = {}
  for index, observation in enumerate(observations):
    if not observation.get("image") or not observation.get("corners"):
      continue
    key = (observation["camera"], observation["image"])
    grouped.setdefault(key, []).append((index, observation))
  for observation in skipped_observations:
    if (
        observation.get("reason") != "marker_quality_rejected"
        or not observation.get("image")
        or not observation.get("corners")):
      continue
    key = (observation["camera"], observation["image"])
    grouped.setdefault(key, []).append((None, observation))

  checks = []
  sequence = {}
  for (camera_name, image_file), entries in grouped.items():
    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image is None:
      raise ValueError(
        "could not read marker image for validation {}".format(image_file)
      )
    camera = calibration["cameras"][camera_name]
    world_to_camera = rig_poses[camera_name] @ world_to_rig
    rotation_vector, _ = cv2.Rodrigues(world_to_camera[:3, :3])
    translation = world_to_camera[:3, 3]
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    distortion = np.asarray(camera["dist"], dtype=np.float64).reshape(-1)
    capture = entries[0][1]["capture"]
    sequence[camera_name] = sequence.get(camera_name, 0) + 1
    output_name = "{:03d}_{}.jpg".format(
      sequence[camera_name],
      "".join(
        value if value.isalnum() or value in "-_."
        else "_"
        for value in str(capture)
      )
    )
    relative_path = (
      Path("check") / camera_name / output_name
    )
    output_path = Path(destination) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    point_records = []

    for index, observation in entries:
      corners = np.asarray(
        observation["corners"], dtype=np.float64
      ).reshape(4, 2)
      detected = np.asarray(
        observation["image_point"], dtype=np.float64
      )
      if index is None:
        color = (0, 165, 255)
        polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
        detected_pixel = tuple(np.rint(detected).astype(int))
        cv2.circle(
          image, detected_pixel, 7, color, 2, cv2.LINE_AA
        )
        quality = observation["quality"]
        angle = quality.get("view_angle_deg")
        angle_label = (
          "unknown" if angle is None else "{:.1f}".format(angle)
        )
        label = "id={} {} quality rejected edge={:.1f}px angle={}".format(
          observation.get("marker_id"),
          observation.get("occurrence", ""),
          quality["min_edge_px"],
          angle_label
        )
        cv2.putText(
          image, label,
          (
            max(0, detected_pixel[0] + 10),
            max(20, detected_pixel[1] - 10)
          ),
          cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )
        point_records.append({
          "marker_id": observation.get("marker_id"),
          "occurrence": observation.get("occurrence"),
          "world_point": observation["world_point"],
          "detected_image_point": detected.tolist(),
          "quality_rejected": True,
          "quality": quality
        })
        continue
      projected, _ = cv2.projectPoints(
        np.asarray(
          observation["world_point"], dtype=np.float64
        ).reshape(1, 3),
        rotation_vector,
        translation,
        intrinsic,
        distortion
      )
      projected = projected.reshape(2)
      is_inlier = bool(inlier_mask[index])
      color = (0, 200, 0) if is_inlier else (0, 0, 255)
      polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
      cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
      detected_pixel = tuple(np.rint(detected).astype(int))
      projected_pixel = tuple(np.rint(projected).astype(int))
      cv2.circle(
        image, detected_pixel, 7, (0, 255, 255), 2, cv2.LINE_AA
      )
      cv2.drawMarker(
        image, projected_pixel, (255, 0, 255),
        cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA
      )
      cv2.line(
        image, detected_pixel, projected_pixel, color, 2, cv2.LINE_AA
      )
      marker_name = "id={}".format(observation.get("marker_id"))
      if observation.get("occurrence"):
        marker_name += " {}".format(observation["occurrence"])
      label = "{} err={:.2f}px {}".format(
        marker_name, errors[index],
        "inlier" if is_inlier else "outlier"
      )
      cv2.putText(
        image,
        label,
        (
          max(0, detected_pixel[0] + 10),
          max(20, detected_pixel[1] - 10)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
      )
      point_records.append({
        "marker_id": observation.get("marker_id"),
        "occurrence": observation.get("occurrence"),
        "world_point": observation["world_point"].tolist(),
        "detected_image_point": detected.tolist(),
        "projected_image_point": projected.tolist(),
        "reprojection_error_px": float(errors[index]),
        "inlier": is_inlier,
        "quality": observation.get("quality")
      })

    if not cv2.imwrite(str(output_path), image):
      raise RuntimeError(
        "could not write world validation image {}".format(output_path)
      )
    checks.append({
      "capture": capture,
      "camera": camera_name,
      "source_image": str(image_file),
      "validation_image": str(relative_path),
      "points": point_records
    })
  return checks


def multicamera_world_extrinsics(
    calibration_file, correspondence_file, output_file=None,
    ransac_threshold=3.0, loss="soft_l1",
    marker_quality=None, marker_min_edge_px=None,
    marker_min_side_ratio=None, marker_min_area_ratio=None,
    marker_max_view_angle=None
):
  calibration_path = Path(calibration_file).resolve()
  correspondence_path = Path(correspondence_file).resolve()
  calibration = load_calibration_json(calibration_path)
  for name, camera in calibration["cameras"].items():
    if camera.get("model", "standard") != "standard":
      raise ValueError(
        "camera {} is not a standard camera model".format(name)
      )
  marker_quality_overrides = {
    "mode": marker_quality,
    "min_edge_px": marker_min_edge_px,
    "min_side_ratio": marker_min_side_ratio,
    "min_area_ratio": marker_min_area_ratio,
    "max_view_angle_deg": marker_max_view_angle
  }
  observations, skipped, data, mode = (
    load_multicamera_correspondences(
      correspondence_path, calibration["cameras"],
      marker_quality_overrides
    )
  )
  quality_options = _marker_quality_options(
    data, marker_quality_overrides
  )
  (
    world_to_rig, rig_poses, errors, inlier_mask,
    initial_camera, optimization_result
  ) = solve_joint_world(
    calibration, observations, ransac_threshold, loss
  )

  cameras = {}
  for camera_name in calibration["cameras"]:
    world_to_camera = rig_poses[camera_name] @ world_to_rig
    camera_to_world = np.linalg.inv(world_to_camera)
    cameras[camera_name] = {
      "world_to_camera": transform_to_json(world_to_camera),
      "camera_to_world": transform_to_json(camera_to_world),
      "position_world": camera_to_world[:3, 3].tolist()
    }

  inlier_errors = errors[inlier_mask]
  destination = (
    Path(output_file).resolve()
    if output_file is not None
    else calibration_path.parent / "world_extrinsics_multicam.json"
  )
  marker_checks = write_multicamera_check_images(
    calibration,
    rig_poses,
    world_to_rig,
    observations,
    errors,
    inlier_mask,
    destination.parent,
    skipped
  )
  camera_statistics = {}
  for camera_name in calibration["cameras"]:
    indices = [
      index for index, value in enumerate(observations)
      if value["camera"] == camera_name
    ]
    if not indices:
      continue
    camera_errors = errors[indices]
    camera_inliers = inlier_mask[indices]
    camera_statistics[camera_name] = {
      "observation_count": len(indices),
      "inlier_count": int(np.count_nonzero(camera_inliers)),
      "reprojection_rms_px": float(np.sqrt(np.mean(
        camera_errors[camera_inliers] ** 2
      ))) if np.any(camera_inliers) else None,
      "all_points_rms_px": float(np.sqrt(np.mean(camera_errors ** 2))),
      "all_points_max_px": float(np.max(camera_errors))
    }

  joint = {
    "input_mode": mode,
    "observation_count": len(observations),
    "distinct_world_point_count": len({
      tuple(value["world_point"]) for value in observations
    }),
    "participating_cameras": sorted(camera_statistics),
    "inlier_count": int(np.count_nonzero(inlier_mask)),
    "inlier_indices": np.flatnonzero(inlier_mask).tolist(),
    "reprojection_rms_px": float(np.sqrt(np.mean(inlier_errors ** 2))),
    "reprojection_mean_px": float(np.mean(inlier_errors)),
    "reprojection_max_px": float(np.max(inlier_errors)),
    "all_points_rms_px": float(np.sqrt(np.mean(errors ** 2))),
    "all_points_max_px": float(np.max(errors)),
    "point_errors_px": errors.tolist(),
    "initialization_camera": initial_camera,
    "loss": loss,
    "ransac_threshold_px": float(ransac_threshold),
    "optimization_success": bool(optimization_result.success),
    "camera_statistics": camera_statistics,
    "skipped_observations": skipped
  }
  if mode == "marker_centers":
    quality_rejected = [
      value for value in skipped
      if value.get("reason") == "marker_quality_rejected"
    ]
    quality_warnings = [
      value for value in observations
      if value.get("quality", {}).get("status") == "warning"
    ]
    joint["marker_quality"] = {
      "settings": quality_options,
      "detected_observation_count": (
        len(observations) + len(quality_rejected)
      ),
      "rejected_count": len(quality_rejected),
      "warning_count": len(quality_warnings)
    }
  if marker_checks:
    joint["marker_checks"] = marker_checks
  output = {
    "convention": (
      "x_camera = R_world_to_camera * x_world + T_world_to_camera"
    ),
    "method": "joint_multicamera_world_anchor",
    "calibration": str(calibration_path),
    "correspondences": str(correspondence_path),
    "world_units": data.get("world_units", "meters"),
    "rig": {
      "world_to_rig": transform_to_json(world_to_rig),
      "rig_frame": (
        "the common base frame resolved from calibration camera_poses"
      )
    },
    "joint": joint,
    "cameras": cameras
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
  )
  return output, destination


@dataclass
class Worldmulti:
  """Jointly anchor a fixed multi-camera rig to measured world points."""

  calibration: str
  correspondences: str
  output: Optional[str] = None
  ransac_threshold: float = 3.0
  loss: str = "soft_l1"
  marker_quality: Optional[str] = None
  marker_min_edge_px: Optional[float] = None
  marker_min_side_ratio: Optional[float] = None
  marker_min_area_ratio: Optional[float] = None
  marker_max_view_angle: Optional[float] = None

  def execute(self):
    result, output_path = multicamera_world_extrinsics(
      self.calibration,
      self.correspondences,
      self.output,
      self.ransac_threshold,
      self.loss,
      self.marker_quality,
      self.marker_min_edge_px,
      self.marker_min_side_ratio,
      self.marker_min_area_ratio,
      self.marker_max_view_angle
    )
    print(json.dumps(result["joint"], indent=2))
    print("Saved joint world extrinsics to {}".format(output_path))


if __name__ == "__main__":
  run_with(Worldmulti)
