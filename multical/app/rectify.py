"""Generate OpenCV stereo-rectification data from a Multical calibration."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from simple_parsing.helpers import list_field

from multical.config.arguments import run_with
from multical.io.calibration_utils import (
  camera_pose_matrices,
  camera_to_camera_transform,
  load_calibration_json,
  transform_to_json
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_pair(pair):
  parts = pair.split(":", 1)
  if len(parts) != 2 or not all(parts):
    raise ValueError(
      "invalid pair {!r}; use LEFT:RIGHT, for example C1:C2".format(pair)
    )
  return parts[0], parts[1]


def pair_filename(left, right):
  safe = re.compile(r"[^A-Za-z0-9_.-]")
  return "{}_{}".format(safe.sub("_", left), safe.sub("_", right))


def camera_parameters(calibration, camera_name):
  if camera_name not in calibration["cameras"]:
    raise ValueError("camera {} not found in calibration".format(camera_name))
  camera = calibration["cameras"][camera_name]
  if camera.get("model", "standard") != "standard":
    raise ValueError(
      "stereo rectification currently supports the standard camera model"
    )
  return (
    np.asarray(camera["K"], dtype=np.float64),
    np.asarray(camera["dist"], dtype=np.float64).reshape(-1),
    tuple(int(value) for value in camera["image_size"])
  )


def skew(vector):
  x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
  return np.array([
    [0.0, -z, y],
    [z, 0.0, -x],
    [-y, x, 0.0]
  ], dtype=np.float64)


def epipolar_matrices(left_k, right_k, left_to_right):
  """Return E and F for x_right = R * x_left + T."""
  rotation = left_to_right[:3, :3]
  translation = left_to_right[:3, 3]
  essential = skew(translation) @ rotation
  fundamental = (
    np.linalg.inv(right_k).T @ essential @ np.linalg.inv(left_k)
  )
  return essential, fundamental


def rectify_pair(calibration, camera_poses, left, right, alpha=-1.0):
  left_k, left_dist, left_size = camera_parameters(calibration, left)
  right_k, right_dist, right_size = camera_parameters(calibration, right)
  if left_size != right_size:
    raise ValueError(
      "{} and {} have different image sizes {} and {}".format(
        left, right, left_size, right_size
      )
    )

  left_to_right = camera_to_camera_transform(
    camera_poses, left, right
  )
  flags = cv2.CALIB_ZERO_DISPARITY
  left_r, right_r, left_p, right_p, q, left_roi, right_roi = (
    cv2.stereoRectify(
      left_k,
      left_dist,
      right_k,
      right_dist,
      left_size,
      left_to_right[:3, :3],
      left_to_right[:3, 3],
      flags=flags,
      alpha=float(alpha)
    )
  )
  left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    left_k,
    left_dist,
    left_r,
    left_p[:, :3],
    left_size,
    cv2.CV_32FC1
  )
  right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    right_k,
    right_dist,
    right_r,
    right_p[:, :3],
    right_size,
    cv2.CV_32FC1
  )
  essential, fundamental = epipolar_matrices(
    left_k, right_k, left_to_right
  )
  parameters = {
    "left": left,
    "right": right,
    "image_size": list(left_size),
    "alpha": float(alpha),
    "left_to_right": transform_to_json(left_to_right),
    "baseline": float(np.linalg.norm(left_to_right[:3, 3])),
    "E": essential.tolist(),
    "F": fundamental.tolist(),
    "R1": left_r.tolist(),
    "R2": right_r.tolist(),
    "P1": left_p.tolist(),
    "P2": right_p.tolist(),
    "Q": q.tolist(),
    "roi1": list(left_roi),
    "roi2": list(right_roi)
  }
  maps = {
    "left_map_x": left_map_x,
    "left_map_y": left_map_y,
    "right_map_x": right_map_x,
    "right_map_y": right_map_y
  }
  return parameters, maps


def natural_sort_key(value):
  return [
    (1, int(part)) if part.isdigit() else (0, part.lower())
    for part in re.split(r"(\d+)", str(value))
  ]


def find_sample_images(image_path, left, right, frame=None):
  root = Path(image_path)
  left_files = {
    path.name: path
    for path in (root / left).iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  }
  right_files = {
    path.name: path
    for path in (root / right).iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  }
  common = sorted(
    set(left_files) & set(right_files), key=natural_sort_key
  )
  if not common:
    raise ValueError(
      "no matching images found for {} and {} under {}".format(
        left, right, root
      )
    )
  if frame is None:
    selected = common[0]
  else:
    matches = [
      name for name in common
      if name == frame or Path(name).stem == Path(frame).stem
    ]
    if not matches:
      raise ValueError(
        "frame {} is not shared by {} and {}".format(frame, left, right)
      )
    selected = matches[0]
  return selected, left_files[selected], right_files[selected]


def create_preview(left_image, right_image, maps):
  left_rectified = cv2.remap(
    left_image,
    maps["left_map_x"],
    maps["left_map_y"],
    cv2.INTER_LINEAR
  )
  right_rectified = cv2.remap(
    right_image,
    maps["right_map_x"],
    maps["right_map_y"],
    cv2.INTER_LINEAR
  )
  preview = np.hstack([left_rectified, right_rectified])
  height, width = preview.shape[:2]
  for y in range(40, height, 60):
    cv2.line(preview, (0, y), (width - 1, y), (0, 220, 0), 1)
  return preview, left_rectified, right_rectified


def generate_rectification(
    calibration_file, pairs, output_path=None, alpha=-1.0,
    image_path=None, frame=None):
  calibration_path = Path(calibration_file).resolve()
  calibration = load_calibration_json(calibration_path)
  camera_poses = camera_pose_matrices(calibration)
  if not pairs:
    raise ValueError("at least one --pairs LEFT:RIGHT value is required")

  destination = (
    Path(output_path).resolve()
    if output_path is not None
    else calibration_path.parent / "rectification"
  )
  destination.mkdir(parents=True, exist_ok=True)
  result_pairs = {}

  for pair in pairs:
    left, right = parse_pair(pair)
    parameters, maps = rectify_pair(
      calibration, camera_poses, left, right, alpha
    )
    basename = pair_filename(left, right)
    map_path = destination / "{}_maps.npz".format(basename)
    np.savez_compressed(map_path, **maps)
    parameters["maps"] = map_path.name

    if image_path is not None:
      selected, left_path, right_path = find_sample_images(
        image_path, left, right, frame
      )
      left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
      right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
      if left_image is None or right_image is None:
        raise RuntimeError("could not read stereo preview images")
      expected_size = tuple(parameters["image_size"])
      if (
          (left_image.shape[1], left_image.shape[0]) != expected_size or
          (right_image.shape[1], right_image.shape[0]) != expected_size):
        raise ValueError("preview image size does not match calibration")
      preview, left_rectified, right_rectified = create_preview(
        left_image, right_image, maps
      )
      preview_path = destination / "{}_preview.jpg".format(basename)
      left_output = destination / "{}_left.jpg".format(basename)
      right_output = destination / "{}_right.jpg".format(basename)
      cv2.imwrite(str(preview_path), preview)
      cv2.imwrite(str(left_output), left_rectified)
      cv2.imwrite(str(right_output), right_rectified)
      parameters["preview_frame"] = selected
      parameters["preview"] = preview_path.name
      parameters["left_rectified"] = left_output.name
      parameters["right_rectified"] = right_output.name

    result_pairs["{}:{}".format(left, right)] = parameters

  result = {
    "convention": (
      "left_to_right maps x_left coordinates into x_right coordinates"
    ),
    "calibration": str(calibration_path),
    "quality": calibration.get("quality"),
    "pairs": result_pairs
  }
  json_path = destination / "rectification.json"
  json_path.write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
  )
  return result, json_path


@dataclass
class Rectify:
  """Generate stereo rectification for one or more camera pairs."""

  calibration: str
  pairs: List[str] = list_field()
  output_path: Optional[str] = None
  alpha: float = -1.0
  image_path: Optional[str] = None
  frame: Optional[str] = None

  def execute(self):
    result, output_file = generate_rectification(
      self.calibration,
      self.pairs,
      self.output_path,
      self.alpha,
      self.image_path,
      self.frame
    )
    print(json.dumps({
      "pairs": list(result["pairs"]),
      "output": str(output_file)
    }, indent=2))


if __name__ == "__main__":
  run_with(Rectify)
