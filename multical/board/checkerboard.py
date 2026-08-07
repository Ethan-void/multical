from cached_property import cached_property
import cv2
import numpy as np
from pprint import pformat

from multical.board.board import Board
from multical.optimization.parameters import Parameters
from structs.struct import choose, struct, subset

from .common import empty_detection, estimate_pose_points, grid_mesh


class Checkerboard(Parameters, Board):
  """A conventional black-and-white checkerboard.

  ``size`` is the number of inner corners in (columns, rows), matching
  OpenCV's ``findChessboardCorners`` convention.
  """

  def __init__(self, size, square_length, min_points=6, min_rows=2,
      subpix_region=5, use_sb=True, adjusted_points=None):
    self.size = tuple(size)
    self.square_length = square_length
    self.min_points = min_points
    self.min_rows = min_rows
    self.subpix_region = subpix_region
    self.use_sb = use_sb
    self.adjusted_points = choose(adjusted_points, self.points)

  def export(self):
    return struct(
      type='checkerboard',
      size=self.size,
      square_length=self.square_length,
      min_points=self.min_points,
      min_rows=self.min_rows,
      subpix_region=self.subpix_region,
      use_sb=self.use_sb
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    width, height = self.size
    points = np.zeros((width * height, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    return points * self.square_length

  @property
  def num_points(self):
    return self.size[0] * self.size[1]

  @property
  def ids(self):
    return np.arange(self.num_points)

  @cached_property
  def mesh(self):
    # grid_mesh expects a Charuco-style size expressed as square counts,
    # whereas checkerboard ``size`` is expressed as inner-corner counts.
    return grid_mesh(
      self.adjusted_points,
      (self.size[0] + 1, self.size[1] + 1)
    )

  @property
  def size_mm(self):
    # A pattern with W x H inner corners contains (W+1) x (H+1) squares.
    square_length_mm = self.square_length * 1000
    return [int(round((dim + 1) * square_length_mm)) for dim in self.size]

  def draw(self, pixels_mm=1, margin=20):
    square_px = int(round(self.square_length * 1000 * pixels_mm))
    assert square_px > 0, "checkerboard square_length is too small to draw"

    width, height = self.size
    columns, rows = width + 1, height + 1
    margin_px = int(round(margin * pixels_mm))
    image = np.full(
      (rows * square_px + 2 * margin_px,
       columns * square_px + 2 * margin_px),
      255,
      dtype=np.uint8
    )

    for row in range(rows):
      for column in range(columns):
        if (row + column) % 2 == 0:
          x0 = margin_px + column * square_px
          y0 = margin_px + row * square_px
          image[y0:y0 + square_px, x0:x0 + square_px] = 0

    return image

  def detect(self, image):
    if image.ndim == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image

    found = False
    corners = None
    if self.use_sb and hasattr(cv2, 'findChessboardCornersSB'):
      flags = cv2.CALIB_CB_NORMALIZE_IMAGE
      flags |= getattr(cv2, 'CALIB_CB_EXHAUSTIVE', 0)
      flags |= getattr(cv2, 'CALIB_CB_ACCURACY', 0)
      found, corners = cv2.findChessboardCornersSB(gray, self.size, flags=flags)
    else:
      flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
      found, corners = cv2.findChessboardCorners(gray, self.size, flags=flags)
      if found:
        window = (self.subpix_region, self.subpix_region)
        criteria = (
          cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
          30,
          0.0001
        )
        corners = cv2.cornerSubPix(
          gray, corners.astype(np.float32), window, (-1, -1), criteria
        )

    if not found or corners is None:
      return empty_detection

    return struct(
      corners=np.asarray(corners, dtype=np.float32).reshape(-1, 2),
      ids=self.ids.copy()
    )

  def has_min_detections(self, detections):
    if detections.ids.size < self.min_points:
      return False

    width, _ = self.size
    rows = detections.ids // width
    columns = detections.ids % width
    return (
      np.unique(rows).size >= self.min_rows and
      np.unique(columns).size >= self.min_rows
    )

  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)

  @cached_property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return self.copy(adjusted_points=params)

  def copy(self, **kwargs):
    state = self.__getstate__()
    state.update(kwargs)
    return Checkerboard(**state)

  def __getstate__(self):
    return subset(self.__dict__, [
      'size', 'square_length', 'min_points', 'min_rows',
      'subpix_region', 'use_sb', 'adjusted_points'
    ])

  def __str__(self):
    return "Checkerboard " + pformat(self.export())

  def __repr__(self):
    return self.__str__()
