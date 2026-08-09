import numpy as np

from multical.optimization.calibration import (
  Calibration,
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


def test_adjust_outliers_warms_up_before_first_hard_rejection():
  class FakeCalibration:
    def __init__(self):
      self.events = []
      self.optimize = {}
      self.reprojection_error = np.array([0.2, 3.0])
      self.inliers = np.array([True, True])

    def report(self, stage):
      self.events.append(("report", stage))

    def bundle_adjust(self, **kwargs):
      self.events.append(("bundle", kwargs["loss"]))
      return self

    def reject_outliers(self, threshold):
      self.events.append(("reject", threshold))
      return self

    def reject_outlier_frames(self, **kwargs):
      self.events.append(("reject_frames", kwargs["outlier_ratio"]))
      return self

  calibration = FakeCalibration()
  Calibration.adjust_outliers(
    calibration,
    num_adjustments=1,
    select_outliers=lambda errors: 2.0,
    initial_loss="soft_l1",
    warmup_before_outlier_rejection=True,
    loss="linear",
    frame_outlier_ratio=0.8
  )

  action_events = [
    event for event in calibration.events
    if event[0] in {"bundle", "reject", "reject_frames"}
  ]
  assert action_events == [
    ("bundle", "soft_l1"),
    ("reject", 2.0),
    ("reject_frames", 0.8),
    ("bundle", "linear")
  ]
