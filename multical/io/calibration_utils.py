"""Utilities for consuming exported Multical calibration JSON files."""

import json
from pathlib import Path

import numpy as np


def load_calibration_json(filename):
  return json.loads(Path(filename).read_text(encoding="utf-8"))


def transform_from_rt(rotation, translation):
  transform = np.eye(4, dtype=np.float64)
  transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
  transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
  return transform


def transform_from_json(data):
  return transform_from_rt(data["R"], data["T"])


def transform_to_json(transform):
  transform = np.asarray(transform, dtype=np.float64)
  return {
    "R": transform[:3, :3].tolist(),
    "T": transform[:3, 3].tolist(),
    "matrix": transform.tolist()
  }


def camera_pose_matrices(calibration):
  """Return base-to-camera transforms for every exported camera.

  Multical exports a master camera as a plain key and relations such as
  ``C2_to_C1``. Despite the key wording, that stored relation transforms
  points from C1 coordinates into C2 coordinates. This function resolves the
  complete graph and returns transforms sharing one calibration base frame.
  """
  camera_names = list(calibration["cameras"])
  exported = calibration.get("camera_poses", {})
  known = {}
  adjacency = {name: [] for name in camera_names}

  for key, value in exported.items():
    transform = transform_from_json(value)
    if "_to_" not in key:
      if key not in adjacency:
        raise ValueError("unknown camera pose key {}".format(key))
      known[key] = transform
      continue

    source, destination = key.split("_to_", 1)
    if source not in adjacency or destination not in adjacency:
      raise ValueError("unknown camera relation {}".format(key))
    # Stored transform maps destination coordinates to source coordinates.
    adjacency[destination].append((source, transform))
    adjacency[source].append((destination, np.linalg.inv(transform)))

  if not known:
    raise ValueError("calibration contains no absolute/master camera pose")

  queue = list(known)
  while queue:
    current = queue.pop(0)
    current_pose = known[current]
    for destination, current_to_destination in adjacency[current]:
      destination_pose = current_to_destination @ current_pose
      if destination in known:
        if not np.allclose(
            known[destination], destination_pose, atol=1e-6):
          raise ValueError(
            "inconsistent camera pose graph at {}".format(destination)
          )
      else:
        known[destination] = destination_pose
        queue.append(destination)

  missing = set(camera_names) - set(known)
  if missing:
    raise ValueError(
      "camera pose graph is disconnected; missing {}".format(
        ", ".join(sorted(missing))
      )
    )
  return known


def camera_to_camera_transform(camera_poses, source, destination):
  """Transform points from ``source`` camera coordinates to ``destination``."""
  if source not in camera_poses:
    raise ValueError("camera {} not found".format(source))
  if destination not in camera_poses:
    raise ValueError("camera {} not found".format(destination))
  return camera_poses[destination] @ np.linalg.inv(camera_poses[source])
