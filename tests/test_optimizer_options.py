from multical.config.arguments import OptimizerOpts
from multical.config.workspace import optimize


def test_iter_option_controls_adjustment_rounds():
  class FakeWorkspace:
    def calibrate(self, name, **kwargs):
      self.name = name
      self.kwargs = kwargs

  workspace = FakeWorkspace()
  optimize(workspace, OptimizerOpts(iter=7))

  assert workspace.name == "calibration"
  assert workspace.kwargs["num_adjustments"] == 7
  assert workspace.kwargs["initial_loss"] == "soft_l1"
  assert workspace.kwargs["outlier_min_threshold"] == 1.0
  assert workspace.kwargs["outlier_max_threshold"] is None
  assert workspace.kwargs["frame_outlier_ratio"] == 0.8
  assert workspace.kwargs["final_recheck_iterations"] == 2
