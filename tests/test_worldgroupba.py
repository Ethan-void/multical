from pathlib import Path

import cv2
import numpy as np

from multical.app.worldgroupba import (
  _JointProblem,
  _parameters_to_transform,
  _project,
  _resolve_prior_weights,
  _transform_to_parameters,
  _write_final_marker_checks
)


def _camera():
  return {
    "K": [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
    "dist": [0.0, 0.0, 0.0, 0.0, 0.0]
  }


def _translation(x, y, z):
  transform = np.eye(4)
  transform[:3, 3] = [x, y, z]
  return transform


def test_transform_parameter_roundtrip():
  transform = _translation(1.0, -2.0, 3.0)
  rotation, _ = cv2.Rodrigues(
    np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
  )
  transform[:3, :3] = rotation

  assert np.allclose(
    _parameters_to_transform(_transform_to_parameters(transform)),
    transform
  )


def test_joint_problem_combines_charuco_world_and_relative_prior():
  intrinsic = {"cameras": {"cam0": _camera(), "cam1": _camera()}}
  local_cam1 = _translation(1.0, 0.0, 0.0)
  world_to_group = _translation(0.0, 0.0, 5.0)
  frame_pose = _translation(0.0, 0.0, 5.0)
  points = np.asarray([
    [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
    [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]
  ])
  world_points = np.asarray([
    [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]
  ])
  charuco_blocks = []
  world_observations = []
  world_blocks = {}
  for camera_name, local_pose in (
      ("cam0", np.eye(4)), ("cam1", local_cam1)):
    charuco_blocks.append({
      "camera": camera_name,
      "frame_key": (0, 0),
      "points": points,
      "image_points": _project(
        intrinsic["cameras"][camera_name], local_pose @ frame_pose, points
      )
    })
    start = len(world_observations)
    image_points = _project(
      intrinsic["cameras"][camera_name],
      local_pose @ world_to_group,
      world_points
    )
    for point, image_point in zip(world_points, image_points):
      world_observations.append({
        "camera": camera_name,
        "world_point": point,
        "image_point": image_point
      })
    world_blocks[camera_name] = {
      "camera": camera_name,
      "indices": np.arange(start, start + len(world_points)),
      "points": world_points,
      "image_points": image_points
    }
  group = {
    "name": "group01",
    "cameras": ["cam0", "cam1"],
    "reference": "cam0",
    "world_to_group": world_to_group,
    "local_poses": {"cam0": np.eye(4), "cam1": local_cam1},
    "frame_poses": {(0, 0): frame_pose},
    "charuco_blocks": charuco_blocks,
    "world_observations": world_observations,
    "world_blocks": world_blocks
  }
  problem = _JointProblem(
    intrinsic, {}, [group], 1.0, 1.0, 0.1, 0.02,
    {"group01": 5.0}
  )

  assert np.allclose(problem.residuals(problem.initial_parameters), 0.0)
  perturbed = problem.initial_parameters.copy()
  camera_slice = problem.slices[("group01", "camera", "cam1")]
  perturbed[camera_slice.start + 3] += 0.05
  residuals = problem.residuals(perturbed)
  assert np.linalg.norm(residuals) > 0.0
  assert problem.sparsity.shape == (
    problem.residual_count,
    problem.initial_parameters.size
  )


def test_joint_problem_uses_group_specific_prior_weights():
  intrinsic = {
    "cameras": {
      "cam0": _camera(), "cam1": _camera(),
      "cam2": _camera(), "cam3": _camera()
    }
  }

  def group(name, reference, secondary):
    return {
      "name": name,
      "cameras": [reference, secondary],
      "reference": reference,
      "world_to_group": np.eye(4),
      "local_poses": {
        reference: np.eye(4), secondary: _translation(1.0, 0.0, 0.0)
      },
      "frame_poses": {},
      "charuco_blocks": [],
      "world_observations": [],
      "world_blocks": {}
    }

  groups = [
    group("group01", "cam0", "cam1"),
    group("group23", "cam2", "cam3")
  ]
  problem = _JointProblem(
    intrinsic, {}, groups, 1.0, 1.0, 0.1, 0.02,
    {"group01": 2.0, "group23": 5.0}
  )
  perturbed = problem.initial_parameters.copy()
  for name, camera in (("group01", "cam1"), ("group23", "cam3")):
    camera_slice = problem.slices[(name, "camera", camera)]
    perturbed[camera_slice.start + 3] += 0.05
  residuals = problem.residuals(perturbed)
  prior_norms = {
    block["group"]: np.linalg.norm(residuals[block["rows"]])
    for block in problem.residual_blocks if block["kind"] == "prior"
  }

  assert np.isclose(prior_norms["group23"] / prior_norms["group01"], 2.5)


def test_resolve_prior_weights_supports_defaults_and_group_overrides():
  assert _resolve_prior_weights(["group01", "group23"], 5.0) == {
    "group01": 5.0, "group23": 5.0
  }
  assert _resolve_prior_weights(
    ["group01", "group23"], 5.0, [2.0, 5.0]
  ) == {"group01": 2.0, "group23": 5.0}

  try:
    _resolve_prior_weights(["group01", "group23"], 5.0, [2.0])
  except ValueError as error:
    assert "must match group_names" in str(error)
  else:
    raise AssertionError("expected mismatched group weights to fail")


def test_final_marker_checks_use_final_solution_and_separate_directory(
    tmp_path, monkeypatch):
  captured = {}

  def fake_write(
      calibration, rig_poses, world_to_rig, observations,
      errors, inlier_mask, destination, skipped_observations=(),
      check_directory="check"):
    captured.update({
      "calibration": calibration,
      "rig_poses": rig_poses,
      "world_to_rig": world_to_rig,
      "observations": observations,
      "errors": errors,
      "inlier_mask": inlier_mask,
      "destination": destination,
      "skipped_observations": skipped_observations,
      "check_directory": check_directory
    })
    return [{
      "validation_image": "check/final/group01/cam0/001_capture.jpg"
    }]

  monkeypatch.setattr(
    "multical.app.worldgroupba.write_multicamera_check_images",
    fake_write
  )
  intrinsic = {"cameras": {"cam0": _camera()}}
  observations = [{"camera": "cam0"}]
  skipped = [{"camera": "cam0", "reason": "marker_quality_rejected"}]
  world_to_group = _translation(0.0, 0.0, 5.0)
  local_poses = {"cam0": np.eye(4)}
  errors = np.asarray([1.25])
  inliers = np.asarray([True])

  checks = _write_final_marker_checks(
    intrinsic,
    {
      "name": "group01",
      "world_observations": observations,
      "skipped_world_observations": skipped
    },
    {
      "local_poses": local_poses,
      "world_to_group": world_to_group
    },
    errors,
    inliers,
    tmp_path / "world_extrinsic.json"
  )

  assert checks == [{
    "validation_image": "check/final/group01/cam0/001_capture.jpg"
  }]
  assert captured["calibration"] is intrinsic
  assert captured["rig_poses"] is local_poses
  assert captured["world_to_rig"] is world_to_group
  assert captured["observations"] is observations
  assert captured["errors"] is errors
  assert captured["inlier_mask"] is inliers
  assert captured["destination"] == tmp_path
  assert captured["skipped_observations"] is skipped
  assert captured["check_directory"] == (
    Path("check") / "final" / "group01"
  )
