import json

import cv2
import numpy as np
import pytest

from multical.image.camera_layout import camera_poses, render_camera_layout


def test_world_to_camera_inverse_and_optical_axis():
  # A +90 degree Y rotation maps the camera's forward axis to world -X.
  rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
  center = np.array([30, -5, 3])
  data = {"cameras": {"right": {"world_to_camera": {
    "R": rotation.tolist(), "T": (-rotation @ center).tolist()}}}}
  [(name, position, forward)] = camera_poses(data)
  assert name == "right"
  np.testing.assert_allclose(position, center)
  np.testing.assert_allclose(forward, [-1, 0, 0])


def test_layout_exports_png_and_rejects_missing_cameras(tmp_path):
  source = tmp_path / "world_extrinsic.json"
  source.write_text(json.dumps({"cameras": {"vertical": {"world_to_camera": {
    "R": np.eye(3).tolist(), "T": [0, 0, -3]}}}}))
  output = render_camera_layout(source)
  image = cv2.imread(str(output))
  assert image is not None and image.shape[1] > image.shape[0]
  with pytest.raises(ValueError, match="no calibrated cameras"):
    camera_poses({"cameras": {}})
  with pytest.raises(ValueError, match="meters"):
    camera_poses({"world_units": "millimeters", "cameras": {}})


@pytest.mark.parametrize("court,length,width", [
  ("badminton", 13.4, 6.1), ("tennis", 23.77, 10.97),
])
def test_layout_uses_court_dimensions_and_markings(tmp_path, monkeypatch, court, length, width):
  from matplotlib.figure import Figure

  source = tmp_path / "world.json"
  source.write_text(json.dumps({"court": court, "cameras": {
    "cam0": {"world_to_camera": {"R": np.eye(3).tolist(), "T": [1, 4, -3]}}
  }}))
  figures = []
  monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: figures.append(fig))
  render_camera_layout(source)
  ax = figures[0].axes[0]
  assert ax.patches[0].get_width() == pytest.approx(length)
  assert ax.patches[0].get_height() == pytest.approx(width)
  segments = [list(zip(line.get_xdata(), line.get_ydata())) for line in ax.lines]
  if court == "badminton":
    assert [(0.76, -3.05), (0.76, 3.05)] in segments
    assert [(0, 0), (4.72, 0)] in segments
    assert [(4.72, -3.05), (4.72, 3.05)] in segments
  else:
    assert [(5.485, 0), (23.77-5.485, 0)] in segments
  # Explicit pipeline configuration takes precedence over JSON metadata.
  render_camera_layout(source, court="badminton")
  assert figures[-1].axes[0].patches[0].get_width() == 13.4
