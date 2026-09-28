"""Merge independently world-anchored camera groups into one camera system."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

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


def _validate_transform(transform, label):
  transform = np.asarray(transform, dtype=np.float64)
  if transform.shape != (4, 4) or not np.isfinite(transform).all():
    raise ValueError("{} must be a finite 4x4 transform".format(label))
  if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
    raise ValueError("{} has an invalid homogeneous bottom row".format(label))
  rotation = transform[:3, :3]
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
    raise ValueError("{} rotation is not orthonormal".format(label))
  if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
    raise ValueError("{} rotation determinant is not one".format(label))
  return transform


def _rotation_error_deg(first, second):
  delta = first[:3, :3] @ second[:3, :3].T
  cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(cosine)))


def _validate_camera_definition(name, expected, actual):
  for key in ("K", "dist"):
    if key not in expected or key not in actual:
      raise ValueError("camera {} is missing {}".format(name, key))
    if not np.allclose(
        np.asarray(expected[key], dtype=np.float64),
        np.asarray(actual[key], dtype=np.float64),
        rtol=1e-7, atol=1e-9):
      raise ValueError(
        "camera {} {} differs from the intrinsic calibration".format(
          name, key
        )
      )
  for key in ("model", "image_size"):
    if key in expected or key in actual:
      if expected.get(key) != actual.get(key):
        raise ValueError(
          "camera {} {} differs from the intrinsic calibration".format(
            name, key
          )
        )


def _camera_record(world_to_camera):
  camera_to_world = np.linalg.inv(world_to_camera)
  return {
    "world_to_camera": transform_to_json(world_to_camera),
    "camera_to_world": transform_to_json(camera_to_world),
    "position_world": camera_to_world[:3, 3].tolist()
  }


def merge_world_groups(
    intrinsic_file, calibration_files, world_extrinsic_files, output_file,
    calibration_output=None, master=None, group_names=None,
    max_rotation_error_deg=0.01, max_translation_error=1e-4,
    allow_overlap=False
):
  """Merge group world poses and optionally derive one global calibration.

  Groups are disjoint unless allow_overlap is enabled for BA initialization.
  With overlap, the first group supplies each camera initial pose. Each result must
  preserve the relative camera poses exported by that group's calibration.
  """
  calibration_files = list(calibration_files)
  world_extrinsic_files = list(world_extrinsic_files)
  if not calibration_files or len(calibration_files) != len(
      world_extrinsic_files):
    raise ValueError(
      "calibrations and world_extrinsics must have the same non-zero length"
    )
  if group_names is None:
    group_names = ["group{}".format(index) for index in range(
      len(calibration_files)
    )]
  else:
    group_names = list(group_names)
  if len(group_names) != len(calibration_files):
    raise ValueError("group_names must match the number of groups")
  if len(set(group_names)) != len(group_names):
    raise ValueError("group_names must be unique")

  intrinsic_path = Path(intrinsic_file).resolve()
  intrinsic = load_calibration_json(intrinsic_path)
  intrinsic_cameras = intrinsic.get("cameras")
  if not isinstance(intrinsic_cameras, dict) or not intrinsic_cameras:
    raise ValueError("intrinsic calibration contains no cameras")

  merged_cameras = {}
  groups = []
  world_units = None
  seen = set()

  for name, calibration_file, world_file in zip(
      group_names, calibration_files, world_extrinsic_files):
    calibration_path = Path(calibration_file).resolve()
    world_path = Path(world_file).resolve()
    calibration = load_calibration_json(calibration_path)
    world = load_calibration_json(world_path)
    group_cameras = calibration.get("cameras")
    world_cameras = world.get("cameras")
    if not isinstance(group_cameras, dict) or not group_cameras:
      raise ValueError("group {} calibration contains no cameras".format(name))
    if not isinstance(world_cameras, dict) or not world_cameras:
      raise ValueError("group {} world result contains no cameras".format(name))
    if set(group_cameras) != set(world_cameras):
      raise ValueError(
        "group {} calibration and world camera sets differ".format(name)
      )
    duplicate = seen.intersection(group_cameras)
    if duplicate and not allow_overlap:
      raise ValueError(
        "cameras occur in multiple groups: {}".format(
          ", ".join(sorted(duplicate))
        )
      )
    unknown = set(group_cameras) - set(intrinsic_cameras)
    if unknown:
      raise ValueError(
        "group {} contains cameras absent from intrinsic: {}".format(
          name, ", ".join(sorted(unknown))
        )
      )
    if world.get("convention") != _WORLD_CONVENTION:
      raise ValueError("group {} has an unsupported world convention".format(
        name
      ))
    current_units = world.get("world_units", "meters")
    if world_units is None:
      world_units = current_units
    elif current_units != world_units:
      raise ValueError("all groups must use the same world units")

    rig_poses = camera_pose_matrices(calibration)
    reference = next(iter(group_cameras))
    reference_world = _validate_transform(
      transform_from_json(
        world_cameras[reference]["world_to_camera"]
      ),
      "group {} camera {} world_to_camera".format(name, reference)
    )
    consistency = {}
    max_rotation = 0.0
    max_translation = 0.0

    for camera_name, camera in group_cameras.items():
      _validate_camera_definition(
        camera_name, intrinsic_cameras[camera_name], camera
      )
      world_to_camera = _validate_transform(
        transform_from_json(
          world_cameras[camera_name]["world_to_camera"]
        ),
        "group {} camera {} world_to_camera".format(name, camera_name)
      )
      expected_relative = (
        rig_poses[camera_name] @ np.linalg.inv(rig_poses[reference])
      )
      actual_relative = world_to_camera @ np.linalg.inv(reference_world)
      rotation_error = _rotation_error_deg(
        actual_relative, expected_relative
      )
      translation_error = float(np.linalg.norm(
        actual_relative[:3, 3] - expected_relative[:3, 3]
      ))
      consistency[camera_name] = {
        "rotation_error_deg": rotation_error,
        "translation_error": translation_error
      }
      max_rotation = max(max_rotation, rotation_error)
      max_translation = max(max_translation, translation_error)
      merged_cameras.setdefault(camera_name, _camera_record(world_to_camera))

    if max_rotation > float(max_rotation_error_deg):
      raise ValueError(
        "group {} relative rotation changed by {:.6f} deg".format(
          name, max_rotation
        )
      )
    if max_translation > float(max_translation_error):
      raise ValueError(
        "group {} relative translation changed by {:.9f} {}".format(
          name, max_translation, world_units
        )
      )
    groups.append({
      "name": name,
      "calibration": str(calibration_path),
      "world_extrinsics": str(world_path),
      "cameras": list(group_cameras),
      "joint": world.get("joint"),
      "local_consistency": {
        "reference_camera": reference,
        "max_rotation_error_deg": max_rotation,
        "max_translation_error": max_translation,
        "cameras": consistency
      }
    })
    seen.update(group_cameras)

  missing = set(intrinsic_cameras) - seen
  if missing:
    raise ValueError(
      "world groups do not cover intrinsic cameras: {}".format(
        ", ".join(sorted(missing))
      )
    )

  output = {
    "convention": _WORLD_CONVENTION,
    "method": "grouped_world_anchor",
    "world_units": world_units,
    "intrinsic": str(intrinsic_path),
    "groups": groups,
    "cameras": {
      name: merged_cameras[name] for name in intrinsic_cameras
    }
  }
  destination = Path(output_file).resolve()
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
  )

  calibration_destination = None
  if calibration_output is not None:
    master = master or next(iter(intrinsic_cameras))
    if master not in merged_cameras:
      raise ValueError("master camera {} is not in the groups".format(master))
    world_to_master = transform_from_json(
      merged_cameras[master]["world_to_camera"]
    )
    camera_poses = {master: transform_to_json(np.eye(4))}
    for camera_name in intrinsic_cameras:
      if camera_name == master:
        continue
      world_to_camera = transform_from_json(
        merged_cameras[camera_name]["world_to_camera"]
      )
      camera_poses[
        "{}_to_{}".format(camera_name, master)
      ] = transform_to_json(
        world_to_camera @ np.linalg.inv(world_to_master)
      )
    combined = {
      "cameras": intrinsic_cameras,
      "camera_poses": camera_poses,
      "provenance": {
        "method": "derived_from_grouped_world_anchor",
        "world_extrinsics": str(destination),
        "master": master,
        "groups": group_names
      }
    }
    calibration_destination = Path(calibration_output).resolve()
    calibration_destination.parent.mkdir(parents=True, exist_ok=True)
    calibration_destination.write_text(
      json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
  return output, destination, calibration_destination


@dataclass
class Worldgroups:
  """Merge disjoint locally calibrated groups through a shared world frame."""

  intrinsic: str
  calibrations: List[str]
  world_extrinsics: List[str]
  output: str
  calibration_output: Optional[str] = None
  master: Optional[str] = None
  group_names: Optional[List[str]] = None
  max_rotation_error_deg: float = 0.01
  max_translation_error: float = 1e-4
  allow_overlap: bool = False

  def execute(self):
    result, output_path, calibration_path = merge_world_groups(
      self.intrinsic,
      self.calibrations,
      self.world_extrinsics,
      self.output,
      self.calibration_output,
      self.master,
      self.group_names,
      self.max_rotation_error_deg,
      self.max_translation_error,
      self.allow_overlap
    )
    summary = {
      "world_units": result["world_units"],
      "groups": [group["name"] for group in result["groups"]],
      "cameras": list(result["cameras"])
    }
    print(json.dumps(summary, indent=2))
    print("Saved grouped world extrinsics to {}".format(output_path))
    if calibration_path is not None:
      print("Saved derived global calibration to {}".format(
        calibration_path
      ))


if __name__ == "__main__":
  run_with(Worldgroups)
