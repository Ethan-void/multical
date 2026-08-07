from functools import partial, reduce
import operator
from cached_property import cached_property
import numpy as np
import cv2
from structs.numpy import shape

from structs.struct import subset, transpose_structs, transpose_lists

from pprint import pformat

from .transform import rtvec, matrix

from structs.struct import struct
from .optimization.parameters import Parameters

from multiprocessing.pool import ThreadPool
from multical.threading import cpu_count
from multical.io.logging import info

import cv2
from tqdm import tqdm

from structs.struct import split_list


def select_intrinsic_outlier_views(
    error_per_view,
    absolute_limit=0.8,
    mad_scale=3.0,
    max_reject_fraction=0.05,
    min_views=15):
  """Select a bounded set of robust per-view intrinsic outliers."""
  errors = np.asarray(error_per_view, dtype=np.float64).reshape(-1)
  if not np.all(np.isfinite(errors)):
    raise ValueError("intrinsic per-view errors must be finite")
  if absolute_limit is not None and float(absolute_limit) < 0:
    raise ValueError("view_error_limit must be non-negative or None")
  if float(mad_scale) < 0:
    raise ValueError("view_mad_scale must be non-negative")
  if not 0.0 <= float(max_reject_fraction) <= 1.0:
    raise ValueError("max_reject_fraction must be between 0 and 1")
  if int(min_views) < 3:
    raise ValueError("min_views must be at least 3")

  if errors.size == 0:
    return [], {
      'median_RMS': None,
      'MAD_RMS': None,
      'robust_threshold': None,
      'threshold': None,
      'candidate_count': 0,
      'selected_count': 0
    }

  median = float(np.median(errors))
  mad = float(np.median(np.abs(errors - median)))
  robust_threshold = median + float(mad_scale) * 1.4826 * mad
  threshold = (
    robust_threshold
    if absolute_limit is None
    else max(float(absolute_limit), robust_threshold)
  )
  candidates = np.flatnonzero(errors > threshold)
  available = max(int(errors.size) - int(min_views), 0)
  fraction_limit = (
    int(np.ceil(errors.size * float(max_reject_fraction)))
    if max_reject_fraction > 0 else 0
  )
  reject_count = min(candidates.size, available, fraction_limit)
  if reject_count:
    candidates = candidates[
      np.argsort(errors[candidates])[::-1][:reject_count]
    ]
  else:
    candidates = np.array([], dtype=int)

  return [int(index) for index in candidates], {
    'median_RMS': median,
    'MAD_RMS': mad,
    'robust_threshold': float(robust_threshold),
    'threshold': float(threshold),
    'candidate_count': int(np.count_nonzero(errors > threshold)),
    'selected_count': int(candidates.size)
  }



class Camera(Parameters):
  def __init__(self, image_size, intrinsic, dist, model='standard', fix_aspect=False, has_skew=False, error_perview=None, intrinsic_dataset={}):

    assert model in Camera.model,\
        f"unknown camera model {model} options are {list(self.model.keys())}"

    self.model = model

    self.image_size = tuple(image_size)
    self.intrinsic = intrinsic
    self.dist = np.zeros(5) if dist is None else dist
    self.fix_aspect = fix_aspect
    self.has_skew = has_skew
    self.error_perview = error_perview #
    self.intrinsic_dataset = intrinsic_dataset  # Collects views that are used for intrinsic calibration

  model = struct(
      standard=0,
      rational=cv2.CALIB_RATIONAL_MODEL,
      tilted=cv2.CALIB_TILTED_MODEL,
      thin_prism=cv2.CALIB_THIN_PRISM_MODEL
  )

  def __str__(self):
    d = dict(intrinsic=self.intrinsic, dist=self.dist,
             image_size=self.image_size)
    return "Camera " + pformat(d)

  def __repr__(self):
    return self.__str__()

  def approx_eq(self, other):
    assert isinstance(other, Camera)
    return self.image_size == other.image_size \
        and np.allclose(other.intrinsic, self.intrinsic) \
        and np.allclose(other.dist, self.dist)

  @staticmethod
  def flags(model, fix_aspect=False):
    return Camera.model[model] | cv2.CALIB_FIX_ASPECT_RATIO * fix_aspect

  @staticmethod
  def calibrate(boards, intrinsic_error_limit, detections, image_size, max_iter=10, eps=1e-3,
                model='standard', fix_aspect=False, has_skew=False, flags=0,
                max_images=None, min_board_coverage=0.8,
                view_error_limit=0.8, view_mad_scale=3.0,
                filter_iterations=3, max_reject_fraction=0.05,
                min_views=15, selection_seed=0):
    '''
    iteratively selects best images to calculate intrinsic parameters
    '''

    points = calibration_points(boards, detections)
    detected_view_count = len(points.corners)
    detected_image_ids = sorted(set(
      int(value) for value in points.image_ids
    ))
    detected_image_count = len(detected_image_ids)
    points, quality_rejected_image_ids, image_board_coverage = (
      filter_incomplete_images(points, boards, min_board_coverage)
    )
    quality_candidate_view_count = len(points.corners)
    quality_candidate_image_ids = sorted(set(
      int(value) for value in points.image_ids
    ))
    quality_candidate_image_count = len(quality_candidate_image_ids)
    if max_images is not None:
      points = top_detection_coverage(
        points,
        max_images,
        image_size,
        seed=selection_seed
      )
    candidate_view_count = len(points.corners)
    candidate_image_ids = sorted(set(
      int(value) for value in points.image_ids
    ))
    candidate_image_count = len(candidate_image_ids)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    flags = Camera.flags(model, fix_aspect) | flags

    filter_history = []
    rejected_views = []
    filter_iterations = max(int(filter_iterations), 0)
    filter_converged = False
    filter_limit_reached = False

    # Each rejection round is followed by another calibration. The extra
    # final fit ensures the returned K/dist/error describe the filtered data,
    # rather than the observations from immediately before the last removal.
    for filter_round in range(filter_iterations + 1):
      err, K, dist, r, t, _, _, error_perView = cv2.calibrateCameraExtended(points.object_points, points.corners,
                                                                            image_size, None, None, criteria=criteria,
                                                                            flags=flags)

      rejected, filter_stats = select_intrinsic_outlier_views(
        error_perView,
        absolute_limit=view_error_limit,
        mad_scale=view_mad_scale,
        max_reject_fraction=max_reject_fraction,
        min_views=min_views
      )
      can_reject = filter_round < filter_iterations
      applied_rejected = rejected if can_reject else []
      rejected_set = set(int(index) for index in rejected)
      rejected_this_round = [
        {
          'image_id': int(points.image_ids[index]),
          'board_id': int(points.board_offset[index]),
          'RMS': float(np.asarray(error_perView).reshape(-1)[index])
        }
        for index in applied_rejected
      ]
      pending_outliers = [
        {
          'image_id': int(points.image_ids[index]),
          'board_id': int(points.board_offset[index]),
          'RMS': float(np.asarray(error_perView).reshape(-1)[index])
        }
        for index in rejected
        if not can_reject
      ]
      filter_history.append({
        'round': int(filter_round + 1),
        'view_count': int(len(points.corners)),
        'overall_RMS': float(err),
        **filter_stats,
        'rejected_views': rejected_this_round,
        'pending_outlier_views': pending_outliers
      })
      rejected_views.extend(rejected_this_round)
      if filter_stats['candidate_count'] == 0:
        filter_converged = True
        break
      if not can_reject:
        filter_limit_reached = True
        break
      if not rejected_set:
        break

      keep = [
        index for index in range(len(points.corners))
        if index not in rejected_set
      ]
      points = points._map(index_list, keep)

    calibrated_dataset = {
      'board_ids': [int(value) for value in points.board_offset],
      'image_ids': [int(value) for value in points.image_ids],
      'point_counts': [int(len(corners)) for corners in points.corners],
      'detected_view_count': int(detected_view_count),
      'quality_candidate_view_count': int(
        quality_candidate_view_count
      ),
      'candidate_view_count': int(candidate_view_count),
      'detected_image_count': int(detected_image_count),
      'quality_candidate_image_count': int(
        quality_candidate_image_count
      ),
      'candidate_image_count': int(candidate_image_count),
      'detected_image_ids': detected_image_ids,
      'quality_candidate_image_ids': quality_candidate_image_ids,
      'quality_rejected_image_ids': quality_rejected_image_ids,
      'candidate_image_ids': candidate_image_ids,
      'image_board_coverage': image_board_coverage,
      'min_board_coverage': float(min_board_coverage),
      'view_error_limit': (
        float(view_error_limit)
        if view_error_limit is not None else None
      ),
      'view_mad_scale': float(view_mad_scale),
      'filter_iterations': int(filter_iterations),
      'filter_converged': bool(filter_converged),
      'filter_limit_reached': bool(filter_limit_reached),
      'selection_seed': int(selection_seed),
      'filter_history': filter_history,
      'reprojection_rejected_views': rejected_views,
      'reprojection_rejected_image_ids': sorted(set(
        int(view['image_id']) for view in rejected_views
      )),
      'overall_error_limit': float(intrinsic_error_limit),
      'overall_error_limit_met': bool(
        abs(float(err)) < float(intrinsic_error_limit)
      )
    }
    if not calibrated_dataset['filter_converged']:
      info(
        "Intrinsic view filtering stopped with unresolved outliers "
        "(iteration or minimum-view guard reached)."
      )
    if not calibrated_dataset['overall_error_limit_met']:
      info(
        "Intrinsic RMS {:.3f} remains above target {:.3f}, but no "
        "additional robust view outliers can be safely rejected.".format(
          err, intrinsic_error_limit
        )
      )

    return Camera(intrinsic=K, dist=dist, image_size=image_size,
                  model=model, fix_aspect=fix_aspect, has_skew=has_skew,
                  error_perview=error_perView,
                  intrinsic_dataset=calibrated_dataset
                  ), err

  def scale_image(self, factor):
    intrinsic = self.intrinsic.copy()
    intrinsic[:2] *= factor

    return self.copy(intrinsic=intrinsic)

  @cached_property
  def undistort_map(self):
    m, _ = cv2.initUndistortRectifyMap(self.intrinsic, self.dist, None,
                                       self.intrinsic, self.image_size, cv2.CV_32FC2)
    return m

  def undistort_points(self, points):
    undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2), self.intrinsic, self.dist, P=self.intrinsic)
    return undistorted.reshape(*points.shape[:-1], 2)

  def project(self, points):

    projected, _ = cv2.projectPoints(
        cv2.UMat(points.reshape(-1, 1, 3)), np.zeros(3), np.zeros(3), self.intrinsic, self.dist)
    return projected.get().reshape(*points.shape[:-1], 2)

  @cached_property
  def focal_length(self):
    fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
    return np.array([fx, fy])

  @cached_property
  def principle_point(self):
    return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]])

  @cached_property
  def skew(self):
    return self.intrinsic[0, 1] if self.has_skew else 0.0


  @cached_property
  def params(self):
    f = self.focal_length
    if self.fix_aspect:
      f = np.array([f.mean(), f.mean()])

    return struct(
        focal_length=f,
        principle_point=self.principle_point,
        skew = np.array([self.skew]),
        dist=self.dist
    )

  def with_params(self, params):

    f = params.focal_length
    fx, fy = f if not self.fix_aspect else (f[0], f[0])

    px, py = params.principle_point
    skew, = params.skew

    intrinsic = [
        [fx,  skew,   px],
        [0,   fy,  py],
        [0,   0,   1],
    ]

    return self.copy(intrinsic=np.array(intrinsic), dist=params.dist)

  def __getstate__(self):
    return subset(self.__dict__, 
      ['image_size', 'intrinsic', 'dist', 'fix_aspect', 'has_skew', 'model']
    )

  def copy(self, **k):
    d = self.__getstate__()
    d.update(k)
    return Camera(**d)


def board_correspondences(board_id, board, detections):
  non_empty = [d for d in detections if board.has_min_detections(d)]
  img_ids = [id for id, d in enumerate(detections) if board.has_min_detections(d)]
  if len(non_empty) == 0:
    return struct(corners = [], object_points=[], ids=[], board_offset= [], image_ids=[])

  detections = transpose_structs(non_empty)
  return detections._extend(
      object_points=[board.points[ids].astype(np.float32) for ids in detections.ids],
      corners=[corners.astype(np.float32) for corners in detections.corners],
      board_offset=list(np.ones(len(img_ids))*board_id), image_ids=img_ids
  )

def board_frames(board, detections):
  non_empty = [d for d in detections if board.has_min_detections(d)]
  return len(non_empty)


def index_list(xs, indexes):
  # Indexing a uniform list through an object ndarray can expand it into a
  # higher-dimensional object array; ``tolist`` then silently converts each
  # OpenCV-compatible float32 ndarray into nested Python lists.
  return [xs[int(index)] for index in indexes]


def coverage(corners, bins):
  hist, _, _ = np.histogram2d(corners[:, 0], corners[:, 1], bins)
  counts = np.count_nonzero(hist)

  return counts

def image_bins(image_size, approx_bins=10):
  bin_size = min(image_size[0] / approx_bins, image_size[1] / approx_bins)

  return [np.linspace(0, image_size[axis], int(image_size[axis] / bin_size)) 
    for axis in [0, 1]]


def top_detection_coverage(
    detections, k, image_size, approx_bins=10, jitter=0.1, seed=0):
  bins = image_bins(image_size, approx_bins=approx_bins)
  bin_jitter = jitter * (approx_bins * approx_bins)
  rng = np.random.default_rng(seed)

  sizes = [-coverage(corners, bins) + rng.normal(0, bin_jitter)
    for corners in detections.corners]

  sorted = detections._map(index_list, np.argsort(sizes))
  return sorted._map(lambda xs: xs[:k])

def calibration_points(boards, detections):

  board_detections = transpose_lists(detections)
  board_points = [board_correspondences(board_id, board, detections) for board_id, (board, detections)
                  in enumerate(zip(boards, board_detections))]

  return reduce(operator.add, board_points)


def filter_incomplete_images(points, boards, min_board_coverage):
  """Remove whole images whose best detected board is too incomplete."""
  threshold = float(min_board_coverage)
  if not 0.0 <= threshold <= 1.0:
    raise ValueError("min_board_coverage must be between 0 and 1")
  if len(points.corners) == 0:
    raise ValueError("no valid board detections for intrinsic calibration")

  best_coverage = {}
  for ids, board_id, image_id in zip(
      points.ids, points.board_offset, points.image_ids):
    board_point_count = len(boards[int(board_id)].points)
    coverage_value = (
      float(len(np.unique(ids))) / float(board_point_count)
      if board_point_count else 0.0
    )
    image_id = int(image_id)
    best_coverage[image_id] = max(
      coverage_value, best_coverage.get(image_id, 0.0)
    )

  accepted = {
    image_id for image_id, value in best_coverage.items()
    if value >= threshold
  }
  rejected = sorted(set(best_coverage) - accepted)
  if not accepted:
    raise ValueError(
      "all intrinsic images were rejected by min_board_coverage={:.3f}; "
      "lower --intrinsic_min_board_coverage or improve the images".format(
        threshold
      )
    )
  indices = [
    index for index, image_id in enumerate(points.image_ids)
    if int(image_id) in accepted
  ]
  filtered = points._map(index_list, indices)
  coverage_report = {
    str(image_id): float(best_coverage[image_id])
    for image_id in sorted(best_coverage)
  }
  return filtered, rejected, coverage_report


def calibrate_cameras(boards, points, image_sizes, intrinsic_error_limit, **kwargs):

  with ThreadPool() as pool:
    f = partial(Camera.calibrate, boards, intrinsic_error_limit, **kwargs)
    return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


def undistort_image(args):
  image, undistort_map = args
  return cv2.remap(image, undistort_map, None, cv2.INTER_CUBIC)


def undistort_images(images, cameras, j=cpu_count(), chunksize=4):
  with ThreadPool(processes=j) as pool:
    image_pairs = [(image, camera.undistort_map)
                   for camera, cam_images in zip(cameras, images)
                   for image in cam_images]

    loader = pool.imap(undistort_image, image_pairs, chunksize=chunksize)
    undistorted = list(tqdm(loader, total=len(image_pairs)))

    return split_list(undistorted, [len(i) for i in images])


def stereo_calibrate(cameras, matches, max_iter=60, eps=1e-6,
                     fix_aspect=False, fix_intrinsic=True):

  left, right = cameras
  criteria = (cv2.TERM_CRITERIA_EPS +
              cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

  assert left.image_size == right.image_size
  assert left.model == right.model

  model = left.model
  image_size = left.image_size

  flags = (Camera.flags(model, fix_aspect) | cv2.CALIB_USE_INTRINSIC_GUESS |
           cv2.CALIB_FIX_INTRINSIC * fix_intrinsic)

  err, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
      matches.object_points, matches.points1, matches.points2,
      left.intrinsic, left.dist,
      right.intrinsic, right.dist,
      image_size, criteria=criteria, flags=flags)

  left = Camera(dist=d1, intrinsic=K1, image_size=image_size, model=model)
  right = Camera(dist=d2, intrinsic=K2, image_size=image_size, model=model)

  return left, right, matrix.join(R, T.flatten()), err
