"""Reconstruct synchronized image observations in world coordinates."""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from scipy import optimize
from simple_parsing import choice

from multical.config.arguments import run_with
from multical.io.calibration_utils import (
  load_calibration_json,
  transform_from_json
)


def load_json_or_yaml(filename):
  path = Path(filename)
  text = path.read_text(encoding="utf-8")
  return (
    json.loads(text)
    if path.suffix.lower() == ".json"
    else yaml.safe_load(text)
  )


def load_observation_frames(filename):
  data = load_json_or_yaml(filename)
  frames = data.get("frames") if isinstance(data, dict) else data
  if not isinstance(frames, list):
    raise ValueError("observations must contain a frames list")
  return data, frames


def parse_observation(value):
  if isinstance(value, dict):
    point = (
      value.get("point")
      or value.get("pixel")
      or value.get("uv")
    )
    confidence = float(value.get("confidence", 1.0))
  else:
    point = value
    confidence = 1.0
  point = np.asarray(point, dtype=np.float64)
  if point.shape != (2,) or not np.isfinite(point).all():
    raise ValueError("each camera observation must be a finite [u, v]")
  if not np.isfinite(confidence) or confidence <= 0:
    raise ValueError("observation confidence must be positive")
  return point, confidence


def build_camera_models(calibration, world_extrinsics):
  models = {}
  world_cameras = world_extrinsics.get("cameras", {})
  for camera_name, camera in calibration["cameras"].items():
    if camera_name not in world_cameras:
      continue
    if camera.get("model", "standard") != "standard":
      raise ValueError(
        "triangulation currently supports the standard camera model"
      )
    world_to_camera = transform_from_json(
      world_cameras[camera_name]["world_to_camera"]
    )
    models[camera_name] = {
      "K": np.asarray(camera["K"], dtype=np.float64),
      "dist": np.asarray(camera["dist"], dtype=np.float64).reshape(-1),
      "world_to_camera": world_to_camera,
      "projection": world_to_camera[:3],
      "position_world": np.linalg.inv(world_to_camera)[:3, 3]
    }
  return models


def normalized_point(model, pixel):
  normalized = cv2.undistortPoints(
    np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
    model["K"],
    model["dist"]
  )
  return normalized.reshape(2)


def triangulate_dlt(camera_names, observations, models):
  rows = []
  for camera_name in camera_names:
    pixel, confidence = observations[camera_name]
    x, y = normalized_point(models[camera_name], pixel)
    projection = models[camera_name]["projection"]
    weight = np.sqrt(confidence)
    rows.append(weight * (x * projection[2] - projection[0]))
    rows.append(weight * (y * projection[2] - projection[1]))
  _, _, vectors = np.linalg.svd(np.asarray(rows))
  homogeneous = vectors[-1]
  if abs(homogeneous[3]) < 1e-12:
    raise ValueError("triangulation produced a point at infinity")
  point = homogeneous[:3] / homogeneous[3]
  if not np.isfinite(point).all():
    raise ValueError("triangulation produced a non-finite point")
  return point


def project_world_point(model, point_world):
  world_to_camera = model["world_to_camera"]
  rotation = world_to_camera[:3, :3]
  translation = world_to_camera[:3, 3]
  rotation_vector, _ = cv2.Rodrigues(rotation)
  projected, _ = cv2.projectPoints(
    np.asarray(point_world, dtype=np.float64).reshape(1, 3),
    rotation_vector,
    translation,
    model["K"],
    model["dist"]
  )
  depth = float(
    (rotation @ np.asarray(point_world) + translation)[2]
  )
  return projected.reshape(2), depth


def reprojection_errors(point_world, camera_names, observations, models):
  errors = {}
  depths = {}
  for camera_name in camera_names:
    projected, depth = project_world_point(
      models[camera_name], point_world
    )
    errors[camera_name] = float(np.linalg.norm(
      projected - observations[camera_name][0]
    ))
    depths[camera_name] = depth
  return errors, depths


def reprojection_rms(errors, camera_names):
  return float(np.sqrt(np.mean([
    errors[name] ** 2 for name in camera_names
  ])))


def refine_point_nonlinear(
    initial_point, camera_names, observations, models,
    loss="linear", max_iterations=50):
  """Minimize pixel reprojection error while keeping cameras fixed."""
  initial_point = np.asarray(initial_point, dtype=np.float64)
  initial_errors, _ = reprojection_errors(
    initial_point, camera_names, observations, models
  )
  initial_rms = reprojection_rms(initial_errors, camera_names)

  def residuals(point_world):
    values = []
    for camera_name in camera_names:
      projected, _ = project_world_point(
        models[camera_name], point_world
      )
      observed, confidence = observations[camera_name]
      values.extend(
        np.sqrt(confidence) * (projected - observed)
      )
    return np.asarray(values, dtype=np.float64)

  try:
    result = optimize.least_squares(
      residuals,
      initial_point,
      method="trf",
      loss=loss,
      max_nfev=max_iterations
    )
  except (ValueError, FloatingPointError, np.linalg.LinAlgError):
    return initial_point, {
      "enabled": True,
      "applied": False,
      "reason": "optimizer failed",
      "initial_reprojection_rms_px": initial_rms,
      "final_reprojection_rms_px": initial_rms
    }

  candidate = np.asarray(result.x, dtype=np.float64)
  if not result.success or not np.isfinite(candidate).all():
    return initial_point, {
      "enabled": True,
      "applied": False,
      "reason": result.message,
      "initial_reprojection_rms_px": initial_rms,
      "final_reprojection_rms_px": initial_rms
    }

  candidate_errors, candidate_depths = reprojection_errors(
    candidate, camera_names, observations, models
  )
  candidate_rms = reprojection_rms(candidate_errors, camera_names)
  valid_depths = all(
    candidate_depths[name] > 0 for name in camera_names
  )
  if not valid_depths or candidate_rms > initial_rms + 1e-12:
    return initial_point, {
      "enabled": True,
      "applied": False,
      "reason": (
        "negative depth" if not valid_depths
        else "reprojection RMS did not improve"
      ),
      "initial_reprojection_rms_px": initial_rms,
      "final_reprojection_rms_px": initial_rms
    }

  return candidate, {
    "enabled": True,
    "applied": True,
    "loss": loss,
    "iterations": int(result.nfev),
    "initial_reprojection_rms_px": initial_rms,
    "final_reprojection_rms_px": candidate_rms,
    "improvement_px": initial_rms - candidate_rms
  }


def maximum_ray_angle_degrees(
    point_world, camera_names, models):
  angles = []
  point_world = np.asarray(point_world)
  for first, second in itertools.combinations(camera_names, 2):
    first_ray = point_world - models[first]["position_world"]
    second_ray = point_world - models[second]["position_world"]
    first_ray /= np.linalg.norm(first_ray)
    second_ray /= np.linalg.norm(second_ray)
    cosine = np.clip(np.dot(first_ray, second_ray), -1.0, 1.0)
    angles.append(float(np.degrees(np.arccos(cosine))))
  return max(angles) if angles else 0.0


def failed_result(frame, reason, cameras):
  result = {
    "frame": frame.get("frame"),
    "status": "failed",
    "reason": reason,
    "cameras_available": cameras
  }
  if "timestamp" in frame:
    result["timestamp"] = frame["timestamp"]
  return result


def triangulate_frame(
    frame, models, reprojection_threshold=3.0,
    min_ray_angle_deg=1.0, refine=True,
    refine_loss="linear", refine_max_iterations=50):
  raw_observations = (
    frame.get("observations") or frame.get("points")
  )
  if not isinstance(raw_observations, dict):
    return failed_result(frame, "missing observations", [])

  observations = {}
  invalid_cameras = []
  for camera_name, value in raw_observations.items():
    if camera_name not in models:
      invalid_cameras.append(camera_name)
      continue
    observations[camera_name] = parse_observation(value)

  camera_names = list(observations)
  if len(camera_names) < 2:
    return failed_result(
      frame, "fewer than two calibrated camera observations", camera_names
    )

  candidates = []
  for first, second in itertools.combinations(camera_names, 2):
    pair = [first, second]
    try:
      point = triangulate_dlt(pair, observations, models)
    except (ValueError, np.linalg.LinAlgError):
      continue
    errors, depths = reprojection_errors(
      point, camera_names, observations, models
    )
    inliers = [
      name for name in camera_names
      if errors[name] <= reprojection_threshold and depths[name] > 0
    ]
    if len(inliers) < 2:
      continue
    inlier_rms = np.sqrt(np.mean([
      errors[name] ** 2 for name in inliers
    ]))
    inlier_confidence = sum(
      observations[name][1] for name in inliers
    )
    capped_total_error = sum(
      min(errors[name], reprojection_threshold * 10)
      for name in camera_names
    )
    score = (
      len(inliers),
      inlier_confidence,
      -float(inlier_rms),
      -float(capped_total_error)
    )
    candidates.append((score, inliers))

  if not candidates:
    return failed_result(
      frame, "no geometrically valid camera pair", camera_names
    )

  _, used_cameras = max(candidates, key=lambda candidate: candidate[0])
  point = None
  for _ in range(4):
    try:
      point = triangulate_dlt(used_cameras, observations, models)
    except (ValueError, np.linalg.LinAlgError):
      return failed_result(
        frame, "multi-view triangulation failed", used_cameras
      )
    errors, depths = reprojection_errors(
      point, camera_names, observations, models
    )
    refined = [
      name for name in camera_names
      if errors[name] <= reprojection_threshold and depths[name] > 0
    ]
    if len(refined) < 2:
      return failed_result(
        frame, "fewer than two inliers after refinement", refined
      )
    if refined == used_cameras:
      break
    used_cameras = refined

  dlt_point = np.asarray(point).copy()
  dlt_errors, _ = reprojection_errors(
    dlt_point, used_cameras, observations, models
  )
  dlt_rms = reprojection_rms(dlt_errors, used_cameras)
  if refine:
    point, refinement = refine_point_nonlinear(
      dlt_point,
      used_cameras,
      observations,
      models,
      loss=refine_loss,
      max_iterations=refine_max_iterations
    )
  else:
    refinement = {
      "enabled": False,
      "applied": False,
      "initial_reprojection_rms_px": dlt_rms,
      "final_reprojection_rms_px": dlt_rms
    }

  errors, depths = reprojection_errors(
    point, camera_names, observations, models
  )
  invalid_refined_point = (
    any(depths[name] <= 0 for name in used_cameras)
    or any(
      errors[name] > reprojection_threshold
      for name in used_cameras
    )
  )
  if invalid_refined_point:
    point = dlt_point
    errors, depths = reprojection_errors(
      point, camera_names, observations, models
    )
    refinement.update({
      "applied": False,
      "reason": "refined point failed geometric validation",
      "final_reprojection_rms_px": dlt_rms
    })

  ray_angle = maximum_ray_angle_degrees(
    point, used_cameras, models
  )
  if ray_angle < min_ray_angle_deg:
    return failed_result(
      frame,
      "triangulation ray angle {:.4f} deg is below threshold".format(
        ray_angle
      ),
      used_cameras
    )

  used_errors = [errors[name] for name in used_cameras]
  result = {
    "frame": frame.get("frame"),
    "status": "ok",
    "point_world": np.asarray(point).tolist(),
    "cameras_used": used_cameras,
    "cameras_rejected": [
      name for name in camera_names if name not in used_cameras
    ] + invalid_cameras,
    "reprojection_errors_px": errors,
    "reprojection_rms_px": float(np.sqrt(np.mean(
      np.square(used_errors)
    ))),
    "reprojection_max_px": float(np.max(used_errors)),
    "max_ray_angle_deg": ray_angle,
    "refinement": refinement
  }
  if "timestamp" in frame:
    result["timestamp"] = frame["timestamp"]
  return result


def triangulate_observations(
    calibration_file, world_extrinsics_file, observations_file,
    output_file=None, reprojection_threshold=3.0,
    min_ray_angle_deg=1.0, refine=True,
    refine_loss="linear", refine_max_iterations=50):
  calibration_path = Path(calibration_file).resolve()
  world_path = Path(world_extrinsics_file).resolve()
  observations_path = Path(observations_file).resolve()
  calibration = load_calibration_json(calibration_path)
  world_extrinsics = load_json_or_yaml(world_path)
  observation_data, frames = load_observation_frames(observations_path)
  models = build_camera_models(calibration, world_extrinsics)
  if len(models) < 2:
    raise ValueError(
      "at least two cameras need calibration and world extrinsics"
    )

  results = [
    triangulate_frame(
      frame,
      models,
      reprojection_threshold,
      min_ray_angle_deg,
      refine,
      refine_loss,
      refine_max_iterations
    )
    for frame in frames
  ]
  successful = [
    result for result in results if result["status"] == "ok"
  ]
  failed = [
    result for result in results if result["status"] != "ok"
  ]
  summary = {
    "frame_count": len(results),
    "reconstructed_count": len(successful),
    "failed_count": len(failed),
    "refined_count": sum(
      bool(result.get("refinement", {}).get("applied"))
      for result in successful
    ),
    "mean_reprojection_rms_px": (
      float(np.mean([
        result["reprojection_rms_px"] for result in successful
      ]))
      if successful else None
    ),
    "max_reprojection_error_px": (
      float(np.max([
        result["reprojection_max_px"] for result in successful
      ]))
      if successful else None
    ),
    "min_ray_angle_deg": (
      float(np.min([
        result["max_ray_angle_deg"] for result in successful
      ]))
      if successful else None
    )
  }
  output = {
    "coordinate_frame": "world",
    "world_units": world_extrinsics.get("world_units", "meters"),
    "convention": world_extrinsics.get("convention"),
    "calibration": str(calibration_path),
    "world_extrinsics": str(world_path),
    "observations": str(observations_path),
    "settings": {
      "reprojection_threshold_px": float(reprojection_threshold),
      "min_ray_angle_deg": float(min_ray_angle_deg),
      "refine": bool(refine),
      "refine_loss": refine_loss,
      "refine_max_iterations": int(refine_max_iterations)
    },
    "summary": summary,
    "frames": results
  }
  if isinstance(observation_data, dict):
    for key in ["sequence", "object", "source"]:
      if key in observation_data:
        output[key] = observation_data[key]

  destination = (
    Path(output_file).resolve()
    if output_file is not None
    else world_path.parent / "triangulation.json"
  )
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
  )
  return output, destination


@dataclass
class Triangulate:
  """Triangulate synchronized camera pixels into world 3D coordinates."""

  calibration: str
  world_extrinsics: str
  observations: str
  output: Optional[str] = None
  reprojection_threshold: float = 3.0
  min_ray_angle_deg: float = 1.0
  refine: bool = True
  refine_loss: str = choice(
    "linear", "soft_l1", "huber", "cauchy", "arctan",
    default="linear"
  )
  refine_max_iterations: int = 50

  def execute(self):
    result, output_path = triangulate_observations(
      self.calibration,
      self.world_extrinsics,
      self.observations,
      self.output,
      self.reprojection_threshold,
      self.min_ray_angle_deg,
      self.refine,
      self.refine_loss,
      self.refine_max_iterations
    )
    print(json.dumps(result["summary"], indent=2))
    print("Saved triangulation to {}".format(output_path))


if __name__ == "__main__":
  run_with(Triangulate)
