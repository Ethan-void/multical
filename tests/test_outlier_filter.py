import numpy as np

from multical.optimization.calibration import (
  reject_outlier_frames_mask,
  select_threshold
)


def test_select_threshold_applies_pixel_floor_and_ceiling():
  errors = np.array([0.1, 0.2, 0.3, 20.0])

  capped = select_threshold(
    quantile=0.75,
    factor=5.0,
    minimum=1.0,
    maximum=10.0
  )
  floored = select_threshold(
    quantile=0.25,
    factor=1.0,
    minimum=1.0,
    maximum=10.0
  )

  assert capped(errors) == 10.0
  assert floored(errors) == 1.0


def test_frame_outlier_rejection_removes_dominated_camera_frame():
  valid = np.ones((1, 2, 1, 4), dtype=bool)
  inliers = valid.copy()
  inliers[0, 0, 0, :3] = False
  inliers[0, 1, 0, 0] = False

  filtered, rejected_frames = reject_outlier_frames_mask(
    valid,
    inliers,
    outlier_ratio=0.75,
    min_points=4
  )

  assert rejected_frames.tolist() == [[True, False]]
  assert not filtered[0, 0].any()
  assert filtered[0, 1].sum() == 3
