import numpy as np

from structs.numpy import struct

from multical.camera import (
  Camera,
  filter_incomplete_images,
  select_intrinsic_outlier_views,
  top_detection_coverage
)


class _Board:
  def __init__(self, point_count):
    self.points = np.zeros((point_count, 3), dtype=np.float32)


def test_filter_incomplete_images_rejects_whole_image():
  points = struct(
    corners=[
      np.zeros((9, 2), dtype=np.float32),
      np.zeros((4, 2), dtype=np.float32),
      np.zeros((8, 2), dtype=np.float32)
    ],
    object_points=[
      np.zeros((9, 3), dtype=np.float32),
      np.zeros((4, 3), dtype=np.float32),
      np.zeros((8, 3), dtype=np.float32)
    ],
    ids=[
      np.arange(9),
      np.arange(4),
      np.arange(8)
    ],
    board_offset=[0, 0, 0],
    image_ids=[0, 1, 2]
  )

  filtered, rejected, coverage = filter_incomplete_images(
    points, [_Board(10)], 0.8
  )

  assert filtered.image_ids == [0, 2]
  assert rejected == [1]
  assert coverage == {"0": 0.9, "1": 0.4, "2": 0.8}


def test_select_intrinsic_outlier_views_uses_absolute_and_robust_limit():
  errors = np.array(
    [0.20] * 18 + [0.95, 1.20],
    dtype=np.float64
  )

  rejected, stats = select_intrinsic_outlier_views(
    errors,
    absolute_limit=0.8,
    mad_scale=3.0,
    max_reject_fraction=0.10,
    min_views=15
  )

  assert rejected == [19, 18]
  assert stats["threshold"] == 0.8


def test_intrinsic_outlier_rejection_is_followed_by_refit(monkeypatch):
  points = struct(
    corners=[
      np.zeros((8, 2), dtype=np.float32) for _ in range(20)
    ],
    object_points=[
      np.zeros((8, 3), dtype=np.float32) for _ in range(20)
    ],
    ids=[np.arange(8) for _ in range(20)],
    board_offset=[0] * 20,
    image_ids=list(range(20))
  )
  fit_view_counts = []

  def fake_calibrate_camera_extended(
      object_points, corners, image_size, _camera, _dist,
      criteria, flags):
    view_count = len(object_points)
    fit_view_counts.append(view_count)
    errors = np.full((view_count, 1), 0.20, dtype=np.float64)
    if view_count == 20:
      errors[-1] = 1.20
    return (
      0.40 if view_count == 20 else 0.20,
      np.eye(3),
      np.zeros((1, 5)),
      [],
      [],
      None,
      None,
      errors
    )

  monkeypatch.setattr(
    "multical.camera.calibration_points",
    lambda boards, detections: points
  )
  monkeypatch.setattr(
    "multical.camera.filter_incomplete_images",
    lambda candidate, boards, threshold: (candidate, [], {})
  )
  monkeypatch.setattr(
    "multical.camera.cv2.calibrateCameraExtended",
    fake_calibrate_camera_extended
  )

  camera, rms = Camera.calibrate(
    [_Board(8)],
    intrinsic_error_limit=0.5,
    detections=[],
    image_size=(640, 480),
    filter_iterations=3,
    max_reject_fraction=0.05,
    min_views=15
  )

  assert fit_view_counts == [20, 19]
  assert rms == 0.20
  assert camera.intrinsic_dataset["image_ids"] == list(range(19))
  assert camera.intrinsic_dataset[
    "reprojection_rejected_image_ids"
  ] == [19]


def test_coverage_selection_is_repeatable_for_the_same_seed():
  detections = struct(
    corners=[
      np.array([[10.0 + index, 10.0]], dtype=np.float32)
      for index in range(20)
    ],
    image_ids=list(range(20))
  )

  first = top_detection_coverage(
    detections, 8, (640, 480), seed=7
  )
  second = top_detection_coverage(
    detections, 8, (640, 480), seed=7
  )

  assert first.image_ids == second.image_ids
