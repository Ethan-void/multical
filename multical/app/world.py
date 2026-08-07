"""Anchor a relative Multical camera rig to a measured world coordinate frame."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from multical.config.arguments import run_with
from multical.io.calibration_utils import (
  camera_pose_matrices,
  camera_to_camera_transform,
  load_calibration_json,
  transform_from_rt,
  transform_to_json
)


def load_correspondences(filename, cameras=None):
  path = Path(filename)
  text = path.read_text(encoding="utf-8")
  if path.suffix.lower() == ".json":
    data = json.loads(text)
  else:
    data = yaml.safe_load(text)

  if "camera" not in data:
    raise ValueError("correspondence file is missing camera")

  marker_observations = None
  if "captures" in data:
    anchor_camera = None
    if cameras is not None:
      if data["camera"] not in cameras:
        raise ValueError(
          "anchor camera {} not found in calibration".format(
            data["camera"]
          )
        )
      anchor_camera = cameras[data["camera"]]
    world_points, image_points, marker_observations = (
      extract_marker_center_correspondences(
        data, path.parent, anchor_camera
      )
    )
  else:
    required = {"world_points", "image_points"}
    missing = required - set(data)
    if missing:
      raise ValueError(
        "correspondence file is missing {}".format(
          ", ".join(sorted(missing))
        )
      )
    world_points = np.asarray(data["world_points"], dtype=np.float64)
    image_points = np.asarray(data["image_points"], dtype=np.float64)

  if (
      world_points.ndim != 2 or world_points.shape[1] != 3 or
      image_points.ndim != 2 or image_points.shape[1] != 2):
    raise ValueError(
      "world_points must be Nx3 and image_points must be Nx2"
    )
  if len(world_points) != len(image_points):
    raise ValueError("world_points and image_points must have equal length")
  if len(world_points) < 4:
    raise ValueError("at least four world/image point pairs are required")
  return (
    data["camera"], world_points, image_points, data,
    marker_observations
  )


def aruco_dictionary(family):
  name = str(family or "6X6_250")
  if not name.startswith("DICT_"):
    name = "DICT_" + name
  candidates = [name, name.upper()]
  for candidate in candidates:
    if hasattr(cv2.aruco, candidate):
      return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, candidate)
      ), candidate
  available = sorted(
    value[5:] for value in dir(cv2.aruco)
    if value.startswith("DICT_")
  )
  raise ValueError(
    "unknown marker_family {}; available families include {}".format(
      family, ", ".join(available)
    )
  )


def diagonal_center(corners):
  """Return the projectively correct intersection of marker diagonals."""
  points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
  homogeneous = np.column_stack([points, np.ones(4)])
  diagonal_02 = np.cross(homogeneous[0], homogeneous[2])
  diagonal_13 = np.cross(homogeneous[1], homogeneous[3])
  center = np.cross(diagonal_02, diagonal_13)
  if abs(center[2]) < 1e-12:
    raise ValueError("marker diagonals do not have a finite intersection")
  return center[:2] / center[2]


def marker_center(corners, camera=None):
  points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
  if camera is None:
    return diagonal_center(points)

  intrinsic = np.asarray(camera["K"], dtype=np.float64)
  distortion = np.asarray(camera["dist"], dtype=np.float64).reshape(-1)
  normalized = cv2.undistortPoints(
    points.reshape(-1, 1, 2),
    intrinsic,
    distortion
  ).reshape(4, 2)
  normalized_center = diagonal_center(normalized)
  distorted_center, _ = cv2.projectPoints(
    np.array([
      [normalized_center[0], normalized_center[1], 1.0]
    ], dtype=np.float64),
    np.zeros(3),
    np.zeros(3),
    intrinsic,
    distortion
  )
  return distorted_center.reshape(2)


def detect_marker_centers(
    image_file, family, camera=None, preserve_duplicates=False):
  image_path = Path(image_file)
  image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError("could not read marker image {}".format(image_path))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  dictionary, dictionary_name = aruco_dictionary(family)
  detected = {}
  # Large close-range markers and small distant markers can require very
  # different adaptive-threshold scales. Detect at several resolutions, map
  # corners back to the original image and retain the highest-resolution hit.
  for scale in (1.0, 0.75, 0.5, 0.25):
    scaled = (
      gray if scale == 1.0 else
      cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    )
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(
      scaled, dictionary, parameters=parameters
    )
    if ids is None:
      continue
    for marker_id, marker_corners in zip(ids.reshape(-1), corners):
      marker_id = int(marker_id)
      if not preserve_duplicates and marker_id in detected:
        continue
      refined_corners = (
        np.asarray(marker_corners, dtype=np.float32).reshape(4, 2) /
        float(scale)
      )
      cv2.cornerSubPix(
        gray,
        refined_corners,
        (5, 5),
        (-1, -1),
        (
          cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
          30,
          0.01
        )
      )
      refined_corners = refined_corners.astype(np.float64)
      detection = {
        "corners": refined_corners,
        "center": marker_center(refined_corners, camera)
      }
      if preserve_duplicates:
        instances = detected.setdefault(marker_id, [])
        # The same physical marker can be found again at another image
        # scale. Keep spatially distinct instances, not scale duplicates.
        if any(
            np.linalg.norm(
              existing["center"] - detection["center"]
            ) < 5.0
            for existing in instances):
          continue
        instances.append(detection)
      else:
        detected[marker_id] = detection
  return image, detected, dictionary_name


def resolve_capture_image(data, capture, config_directory):
  """Resolve a capture image from an override, pattern or common layouts."""
  if capture.get("image"):
    image_path = Path(capture["image"])
    if not image_path.is_absolute():
      image_path = Path(config_directory) / image_path
    if not image_path.is_file():
      raise ValueError(
        "capture image does not exist {}".format(image_path)
      )
    return image_path

  capture_name = capture.get("name", capture.get("frame"))
  if capture_name is None:
    raise ValueError(
      "capture without image must contain name or frame"
    )
  if "image_path" not in data:
    raise ValueError(
      "capture {} has no image; set top-level image_path or a per-capture "
      "image".format(capture_name)
    )

  image_root = Path(data["image_path"])
  if not image_root.is_absolute():
    image_root = Path(config_directory) / image_root
  camera_name = str(data["camera"])
  pattern = data.get("image_pattern")
  if pattern:
    patterned = image_root / str(pattern).format(
      camera=camera_name,
      capture=capture_name,
      frame=capture_name
    )
    if patterned.is_file():
      return patterned
    raise ValueError(
      "capture {} image_pattern resolved to missing file {}".format(
        capture_name, patterned
      )
    )

  extensions = (".jpg", ".jpeg", ".png", ".bmp", ".ppm")
  name_path = Path(str(capture_name))
  names = (
    [name_path.name]
    if name_path.suffix.lower() in extensions else
    [str(capture_name) + extension for extension in extensions]
  )
  candidates = []
  for filename in names:
    candidates.extend([
      image_root / camera_name / filename,
      image_root / str(capture_name) / (
        camera_name if Path(camera_name).suffix else
        camera_name + Path(filename).suffix
      ),
      image_root / filename
    ])
  matches = []
  for candidate in candidates:
    if candidate.is_file() and candidate not in matches:
      matches.append(candidate)
  if len(matches) == 1:
    return matches[0]
  if not matches:
    raise ValueError(
      "could not find image for capture {} under {}; expected "
      "<root>/<camera>/<capture>.<ext>, "
      "<root>/<capture>/<camera>.<ext> or <root>/<capture>.<ext>".format(
        capture_name, image_root
      )
    )
  raise ValueError(
    "capture {} matched multiple images {}; set image_pattern or an "
    "explicit image".format(
      capture_name, ", ".join(str(value) for value in matches)
    )
  )


def _natural_path_key(path):
  return [
    int(part) if part.isdigit() else part.lower()
    for part in re.split(r"(\d+)", str(path))
  ]


def discover_ordered_capture_images(data, config_directory):
  """Discover one image sequence and return it in natural filename order."""
  if "image_path" not in data:
    raise ValueError(
      "unnamed captures require top-level image_path"
    )
  if data.get("image_pattern"):
    raise ValueError(
      "image_pattern requires capture names; remove image_pattern when "
      "mapping captures to images by order"
    )

  image_root = Path(data["image_path"])
  if not image_root.is_absolute():
    image_root = Path(config_directory) / image_root
  if not image_root.is_dir():
    raise ValueError(
      "image_path directory does not exist {}".format(image_root)
    )

  extensions = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
  camera_name = str(data["camera"])

  def image_files(directory):
    if not directory.is_dir():
      return []
    return [
      path for path in directory.iterdir()
      if path.is_file() and path.suffix.lower() in extensions
    ]

  layouts = {
    "camera directory": image_files(image_root / camera_name),
    "flat directory": image_files(image_root),
    "capture directories": []
  }
  for capture_directory in image_root.iterdir():
    if not capture_directory.is_dir():
      continue
    for extension in extensions:
      candidate = capture_directory / (camera_name + extension)
      if candidate.is_file():
        layouts["capture directories"].append(candidate)

  populated = {
    layout: paths for layout, paths in layouts.items() if paths
  }
  if not populated:
    raise ValueError(
      "no calibration images found under {}".format(image_root)
    )
  if len(populated) > 1:
    raise ValueError(
      "image_path contains multiple supported layouts {}; keep one layout "
      "or provide capture names".format(", ".join(populated))
    )
  return sorted(next(iter(populated.values())), key=_natural_path_key)


def extract_marker_center_correspondences(
    data, config_directory, camera=None):
  captures = data.get("captures")
  if not isinstance(captures, list) or not captures:
    raise ValueError("captures must be a non-empty list")
  default_family = data.get("marker_family", "6X6_250")
  world_points = []
  image_points = []
  observations = []
  unnamed = [
    not capture.get("image") and
    capture.get("name") is None and
    capture.get("frame") is None
    for capture in captures
  ]
  ordered_images = None
  if any(unnamed):
    if not all(unnamed):
      raise ValueError(
        "captures cannot mix ordered unnamed entries with named/image "
        "entries"
      )
    ordered_images = discover_ordered_capture_images(
      data, config_directory
    )
    if len(ordered_images) != len(captures):
      raise ValueError(
        "ordered image count {} does not match captures count {}".format(
          len(ordered_images), len(captures)
        )
      )

  for capture_index, capture in enumerate(captures):
    if "markers" not in capture:
      raise ValueError(
        "capture {} must contain markers".format(capture_index)
      )
    image_path = (
      ordered_images[capture_index] if ordered_images is not None else
      resolve_capture_image(data, capture, config_directory)
    )

    markers = capture["markers"]
    if not isinstance(markers, (dict, list)) or not markers:
      raise ValueError(
        "capture {} markers must be a non-empty mapping or list".format(
          capture_index
        )
      )
    marker_specs = []
    marker_entries = (
      list(markers.items()) if isinstance(markers, dict) else
      [(None, marker) for marker in markers]
    )
    for marker_key, marker_value in marker_entries:
      if isinstance(markers, list):
        if not isinstance(marker_value, dict):
          raise ValueError(
            "capture {} marker list entries must be mappings".format(
              capture_index
            )
          )
        if "marker_id" not in marker_value:
          raise ValueError(
            "capture {} marker list entry is missing marker_id".format(
              capture_index
            )
          )
        marker_id = int(marker_value["marker_id"])
      else:
        marker_id = int(marker_key)
      occurrence = None
      if isinstance(marker_value, dict):
        family = marker_value.get("marker_family", default_family)
        occurrence = marker_value.get("occurrence")
        if "world_point" not in marker_value:
          raise ValueError(
            "capture {} marker {} must contain world_point".format(
              capture_index, marker_id
            )
          )
        world_point = marker_value["world_point"]
      else:
        family = default_family
        world_point = marker_value
      if not family:
        raise ValueError(
          "capture {} marker {} is missing marker_family".format(
            capture_index, marker_id
          )
        )
      point = np.asarray(world_point, dtype=np.float64)
      if point.shape != (3,):
        raise ValueError(
          "capture {} marker {} world point must be [X,Y,Z]".format(
            capture_index, marker_id
          )
        )
      if occurrence is not None:
        occurrence = str(occurrence).strip().lower()
        occurrence = {
          "top": "upper",
          "bottom": "lower"
        }.get(occurrence, occurrence)
        if occurrence not in {"upper", "lower"}:
          raise ValueError(
            "capture {} marker {} occurrence must be upper or lower".format(
              capture_index, marker_id
            )
          )
      marker_specs.append({
        "marker_id": marker_id,
        "family": str(family),
        "point": point,
        "occurrence": occurrence
      })

    detections_by_family = {}
    for family in dict.fromkeys(
        spec["family"] for spec in marker_specs):
      _, detected, dictionary_name = detect_marker_centers(
        image_path, family, camera, preserve_duplicates=True
      )
      detections_by_family[family] = (detected, dictionary_name)

    for spec in marker_specs:
      marker_id = spec["marker_id"]
      family = spec["family"]
      point = spec["point"]
      occurrence = spec["occurrence"]
      detected, dictionary_name = detections_by_family[family]
      if marker_id not in detected:
        raise ValueError(
          "capture {} image {} did not detect expected marker {} in {}; "
          "detected IDs for that family are {}".format(
            capture_index,
            image_path,
            marker_id,
            dictionary_name,
            sorted(detected)
          )
        )
      instances = sorted(
        detected[marker_id],
        key=lambda item: float(item["center"][1])
      )
      if occurrence is None:
        if len(instances) != 1:
          raise ValueError(
            "capture {} image {} detected marker {} {} times in {}; "
            "use list-form markers with occurrence upper/lower".format(
              capture_index, image_path, marker_id, len(instances),
              dictionary_name
            )
          )
        detection = instances[0]
      else:
        if len(instances) < 2:
          raise ValueError(
            "capture {} image {} needs upper/lower marker {} in {}, "
            "but detected only {} instance(s)".format(
              capture_index, image_path, marker_id, dictionary_name,
              len(instances)
            )
          )
        detection = (
          instances[0] if occurrence == "upper" else instances[-1]
        )
      world_points.append(point)
      image_points.append(detection["center"])
      observations.append({
        "capture_index": int(capture_index),
        "capture": str(capture.get(
          "name", capture.get("frame", image_path.stem)
        )),
        "image": str(image_path.resolve()),
        "marker_id": marker_id,
        "marker_family": dictionary_name,
        "occurrence": occurrence,
        "world_point": point.tolist(),
        "image_point": detection["center"].tolist(),
        "corners": detection["corners"].tolist()
      })

  return (
    np.asarray(world_points, dtype=np.float64),
    np.asarray(image_points, dtype=np.float64),
    observations
  )


def solve_world_anchor(
    camera, world_points, image_points, ransac_threshold=3.0):
  intrinsic = np.asarray(camera["K"], dtype=np.float64)
  distortion = np.asarray(camera["dist"], dtype=np.float64).reshape(-1)

  success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
    world_points,
    image_points,
    intrinsic,
    distortion,
    iterationsCount=1000,
    reprojectionError=float(ransac_threshold),
    confidence=0.999,
    flags=cv2.SOLVEPNP_EPNP
  )
  if not success or inliers is None or len(inliers) < 4:
    success, rotation_vector, translation = cv2.solvePnP(
      world_points,
      image_points,
      intrinsic,
      distortion,
      flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
      raise RuntimeError("could not solve world-to-camera pose")
    inlier_indices = np.arange(len(world_points), dtype=np.int32)
  else:
    inlier_indices = inliers.reshape(-1)

  if hasattr(cv2, "solvePnPRefineLM") and len(inlier_indices) >= 4:
    rotation_vector, translation = cv2.solvePnPRefineLM(
      world_points[inlier_indices],
      image_points[inlier_indices],
      intrinsic,
      distortion,
      rotation_vector,
      translation
    )

  rotation, _ = cv2.Rodrigues(rotation_vector)
  projected, _ = cv2.projectPoints(
    world_points,
    rotation_vector,
    translation,
    intrinsic,
    distortion
  )
  errors = np.linalg.norm(
    projected.reshape(-1, 2) - image_points, axis=1
  )
  return (
    transform_from_rt(rotation, translation),
    inlier_indices,
    errors
  )


def _safe_filename(value):
  return "".join(
    character if character.isalnum() or character in "-_."
    else "_"
    for character in str(value)
  )


def write_marker_check_images(
    camera,
    world_to_anchor,
    observations,
    inlier_indices,
    destination
):
  if not observations:
    return []

  output_directory = Path(destination) / "world_check"
  output_directory.mkdir(parents=True, exist_ok=True)
  intrinsic = np.asarray(camera["K"], dtype=np.float64)
  distortion = np.asarray(camera["dist"], dtype=np.float64).reshape(-1)
  rotation_vector, _ = cv2.Rodrigues(world_to_anchor[:3, :3])
  translation = world_to_anchor[:3, 3]
  inliers = set(int(value) for value in inlier_indices)

  grouped = {}
  for point_index, observation in enumerate(observations):
    grouped.setdefault(observation["image"], []).append(
      (point_index, observation)
    )

  outputs = []
  for image_index, (image_file, entries) in enumerate(
      grouped.items(), start=1
  ):
    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image is None:
      raise ValueError(
        "could not read marker image for validation {}".format(image_file)
      )
    capture = entries[0][1]["capture"]
    output_name = "{:03d}_{}.jpg".format(
      image_index, _safe_filename(capture)
    )

    point_records = []
    for point_index, observation in entries:
      corners = np.asarray(
        observation["corners"], dtype=np.float64
      ).reshape(4, 2)
      detected = np.asarray(
        observation["image_point"], dtype=np.float64
      )
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
      error = float(np.linalg.norm(projected - detected))
      is_inlier = point_index in inliers
      color = (0, 200, 0) if is_inlier else (0, 0, 255)

      polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
      cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
      detected_pixel = tuple(np.rint(detected).astype(int))
      projected_pixel = tuple(np.rint(projected).astype(int))
      cv2.circle(image, detected_pixel, 7, (0, 255, 255), 2, cv2.LINE_AA)
      cv2.drawMarker(
        image, projected_pixel, (255, 0, 255),
        cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA
      )
      cv2.line(
        image, detected_pixel, projected_pixel, color, 2, cv2.LINE_AA
      )
      marker_name = "id={}".format(observation["marker_id"])
      if observation.get("occurrence"):
        marker_name += " {}".format(observation["occurrence"])
      label = "{} err={:.2f}px {}".format(
        marker_name,
        error,
        "inlier" if is_inlier else "outlier"
      )
      label_position = (
        max(0, detected_pixel[0] + 10),
        max(20, detected_pixel[1] - 10)
      )
      cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
      )
      point_records.append({
        "marker_id": observation["marker_id"],
        "occurrence": observation.get("occurrence"),
        "world_point": observation["world_point"],
        "detected_image_point": detected.tolist(),
        "projected_image_point": projected.tolist(),
        "reprojection_error_px": error,
        "inlier": is_inlier
      })

    output_path = output_directory / output_name
    if not cv2.imwrite(str(output_path), image):
      raise RuntimeError(
        "could not write world validation image {}".format(output_path)
      )
    outputs.append({
      "capture": capture,
      "source_image": str(image_file),
      "validation_image": str(
        Path("world_check") / output_name
      ),
      "points": point_records
    })
  return outputs


def world_extrinsics(
    calibration_file, correspondence_file, output_file=None,
    ransac_threshold=3.0):
  calibration_path = Path(calibration_file).resolve()
  calibration = load_calibration_json(calibration_path)
  (
    anchor_name,
    world_points,
    image_points,
    correspondence_data,
    marker_observations
  ) = (
    load_correspondences(
      correspondence_file, cameras=calibration["cameras"]
    )
  )
  if anchor_name not in calibration["cameras"]:
    raise ValueError(
      "anchor camera {} not found in calibration".format(anchor_name)
    )

  camera_model = calibration["cameras"][anchor_name].get(
    "model", "standard"
  )
  if camera_model != "standard":
    raise ValueError(
      "world anchoring currently supports the standard camera model"
    )

  world_to_anchor, inlier_indices, errors = solve_world_anchor(
    calibration["cameras"][anchor_name],
    world_points,
    image_points,
    ransac_threshold
  )
  relative_poses = camera_pose_matrices(calibration)
  cameras = {}
  for camera_name in calibration["cameras"]:
    anchor_to_camera = camera_to_camera_transform(
      relative_poses, anchor_name, camera_name
    )
    world_to_camera = anchor_to_camera @ world_to_anchor
    camera_to_world = np.linalg.inv(world_to_camera)
    cameras[camera_name] = {
      "world_to_camera": transform_to_json(world_to_camera),
      "camera_to_world": transform_to_json(camera_to_world),
      "position_world": camera_to_world[:3, 3].tolist()
    }

  inlier_errors = errors[inlier_indices]
  destination = (
    Path(output_file).resolve()
    if output_file is not None
    else calibration_path.parent / "world_extrinsics.json"
  )
  marker_checks = write_marker_check_images(
    calibration["cameras"][anchor_name],
    world_to_anchor,
    marker_observations,
    inlier_indices,
    destination.parent
  )

  anchor_output = {
    "camera": anchor_name,
    "point_count": len(world_points),
    "inlier_count": len(inlier_indices),
    "inlier_indices": inlier_indices.tolist(),
    "reprojection_rms_px": float(np.sqrt(np.mean(inlier_errors ** 2))),
    "reprojection_mean_px": float(np.mean(inlier_errors)),
    "reprojection_max_px": float(np.max(inlier_errors)),
    "all_points_rms_px": float(np.sqrt(np.mean(errors ** 2))),
    "all_points_max_px": float(np.max(errors)),
    "point_errors_px": errors.tolist()
  }
  if marker_checks:
    anchor_output["input_mode"] = "marker_centers"
    anchor_output["marker_checks"] = marker_checks
  else:
    anchor_output["input_mode"] = "manual_points"

  output = {
    "convention": (
      "x_camera = R_world_to_camera * x_world + T_world_to_camera"
    ),
    "calibration": str(calibration_path),
    "correspondences": str(Path(correspondence_file).resolve()),
    "world_units": correspondence_data.get("world_units", "meters"),
    "anchor": anchor_output,
    "cameras": cameras
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
  )
  return output, destination


@dataclass
class World:
  """Anchor cameras using manual points or detected marker centers."""

  calibration: str
  correspondences: str
  output: Optional[str] = None
  ransac_threshold: float = 3.0

  def execute(self):
    result, output_path = world_extrinsics(
      self.calibration,
      self.correspondences,
      self.output,
      self.ransac_threshold
    )
    print(json.dumps(result["anchor"], indent=2))
    print("Saved world extrinsics to {}".format(output_path))


if __name__ == "__main__":
  run_with(World)
