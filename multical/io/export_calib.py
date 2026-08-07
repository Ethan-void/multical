import json
from os import path
import numpy as np

from structs.struct import struct, to_dicts, transpose_lists
from multical import graph
from multical.transform import matrix
from multical import tables


def export_camera(camera, quality=None):
  data = struct(
      model = camera.model,
      image_size=camera.image_size, 
      K = camera.intrinsic.tolist(),
      dist = camera.dist.tolist()
  )
  return data if quality is None else data._extend(quality=quality)


def export_cameras(camera_names, cameras, qualities=None):
    qualities = qualities or [None] * len(cameras)
    return {
      k: export_camera(camera, quality)
      for k, camera, quality in zip(camera_names, cameras, qualities)
    }

def export_transform(pose):
    r, t = matrix.split(pose)
    return struct (R = r.tolist(), T=t.tolist())


def export_camera_poses(camera_names, camera_poses):
  return {k : export_transform(pose) 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid) 
      if valid}


def export_relative(camera_names, camera_poses, master):
  assert master in camera_names

  return {k if master == k else f"{k}_to_{master}" : export_transform(pose) 
    for k, pose, valid in zip(camera_names, camera_poses.poses, camera_poses.valid) 
      if valid}


def export_sequential(camera_names, camera_poses):
  transforms = {camera_names[0]: export_transform(np.eye(4))}
  poses = camera_poses.poses

  for i in range(1, len(camera_names)):
    k = f"{camera_names[i]}_to_{camera_names[i - 1]}"
    transforms[k] = export_transform(poses[i] @ np.linalg.inv(poses[i - 1]))
    
  return transforms



def export_poses(pose_table, names=None):
  names = names or [str(i) for i in range(pose_table._size[0])]

  return {i:t.poses.tolist() for i, t in zip(names, pose_table._sequence()) 
    if t.valid}


def export_images(camera_names, filenames):
  return struct(
      rgb = [{camera : image for image, camera in zip(images, camera_names)}
        for images in filenames]
    )


def _rms(errors):
  errors = np.asarray(errors, dtype=np.float64).reshape(-1)
  if errors.size == 0:
    return None
  return float(np.sqrt(np.mean(np.square(errors))))


def export_calibration_quality(calib):
  """Export final bundle-adjustment reprojection quality.

  ``RMS`` is the final inlier reprojection RMS in pixels, matching the first
  value printed by ``Calibration.report``.  The unfiltered value and both
  observation counts are retained so consumers do not need to parse the text
  log or load the Python-only workspace pickle.
  """
  overall = np.asarray(calib.reprojection_error).reshape(-1)
  has_outlier_filter = calib.inlier_mask is not None
  inliers = (
    np.asarray(calib.reprojection_inliers).reshape(-1)
    if has_outlier_filter else overall
  )
  return struct(
    RMS=_rms(inliers),
    RMS_all=_rms(overall),
    unit="px",
    inlier_observation_count=int(inliers.size),
    observation_count=int(overall.size),
    outlier_filter_applied=bool(has_outlier_filter)
  )


EXTRINSIC_QUALITY_THRESHOLDS = struct(
  good=struct(
    min_common_frames=30,
    max_rotation_scatter_deg=0.2,
    max_translation_scatter=0.05
  ),
  caution=struct(
    min_common_frames=20,
    max_rotation_scatter_deg=0.5,
    max_translation_scatter=0.15
  )
)


def _masked_rms(values, mask):
  selected = np.asarray(values, dtype=np.float64)[
    np.asarray(mask, dtype=bool)
  ]
  return _rms(selected)


def _pair_quality_status(
    common_frames, rotation_scatter, translation_scatter):
  if common_frames <= 0:
    return "no_overlap"
  good = EXTRINSIC_QUALITY_THRESHOLDS.good
  caution = EXTRINSIC_QUALITY_THRESHOLDS.caution
  if (
      common_frames >= good.min_common_frames and
      rotation_scatter is not None and
      rotation_scatter <= good.max_rotation_scatter_deg and
      translation_scatter is not None and
      translation_scatter <= good.max_translation_scatter):
    return "good"
  if (
      common_frames >= caution.min_common_frames and
      rotation_scatter is not None and
      rotation_scatter <= caution.max_rotation_scatter_deg and
      translation_scatter is not None and
      translation_scatter <= caution.max_translation_scatter):
    return "caution"
  return "weak"


def _pair_pose_statistics(
    pose_table, first, second, final_transform=None):
  if pose_table is None:
    return struct(
      common_pose_count=None,
      pose_inlier_count=None,
      rotation_scatter_deg=None,
      translation_scatter=None,
      frobenius_scatter=None,
      final_edge_rotation_residual_deg=None,
      final_edge_translation_residual=None,
      final_rotation_residual_rms_deg=None,
      final_translation_residual_rms=None,
      transform=None
    )
  valid = np.asarray(
    pose_table.valid[first] & pose_table.valid[second],
    dtype=bool
  )
  common_pose_count = int(np.count_nonzero(valid))
  if common_pose_count == 0:
    return struct(
      common_pose_count=0,
      pose_inlier_count=0,
      rotation_scatter_deg=None,
      translation_scatter=None,
      frobenius_scatter=None,
      final_edge_rotation_residual_deg=None,
      final_edge_translation_residual=None,
      final_rotation_residual_rms_deg=None,
      final_translation_residual_rms=None,
      transform=None
    )
  poses_first = np.asarray(pose_table.poses[first]).reshape(-1, 4, 4)
  poses_second = np.asarray(pose_table.poses[second]).reshape(-1, 4, 4)
  valid_flat = valid.reshape(-1)
  try:
    transform, inliers = matrix.align_transforms_robust(
      poses_first, poses_second, valid=valid_flat
    )
    inliers = np.asarray(inliers, dtype=bool)
    errors = matrix.pose_errors(
      transform @ poses_first[inliers],
      poses_second[inliers]
    )
    final_edge_rotation_residual_deg = None
    final_edge_translation_residual = None
    final_rotation_residual_rms_deg = None
    final_translation_residual_rms = None
    if final_transform is not None:
      final_edge_error = matrix.pose_errors(
        transform.reshape(1, 4, 4),
        final_transform.reshape(1, 4, 4)
      )
      final_frame_errors = matrix.pose_errors(
        final_transform @ poses_first[inliers],
        poses_second[inliers]
      )
      final_edge_rotation_residual_deg = float(
        final_edge_error.rotation_deg[0]
      )
      final_edge_translation_residual = float(
        final_edge_error.translation[0]
      )
      final_rotation_residual_rms_deg = _rms(
        final_frame_errors.rotation_deg
      )
      final_translation_residual_rms = _rms(
        final_frame_errors.translation
      )
    return struct(
      common_pose_count=common_pose_count,
      pose_inlier_count=int(np.count_nonzero(inliers)),
      rotation_scatter_deg=_rms(errors.rotation_deg),
      translation_scatter=_rms(errors.translation),
      frobenius_scatter=_rms(errors.frobius),
      final_edge_rotation_residual_deg=(
        final_edge_rotation_residual_deg
      ),
      final_edge_translation_residual=(
        final_edge_translation_residual
      ),
      final_rotation_residual_rms_deg=(
        final_rotation_residual_rms_deg
      ),
      final_translation_residual_rms=final_translation_residual_rms,
      transform=transform
    )
  except (ValueError, np.linalg.LinAlgError):
    return struct(
      common_pose_count=common_pose_count,
      pose_inlier_count=None,
      rotation_scatter_deg=None,
      translation_scatter=None,
      frobenius_scatter=None,
      final_edge_rotation_residual_deg=None,
      final_edge_translation_residual=None,
      final_rotation_residual_rms_deg=None,
      final_translation_residual_rms=None,
      transform=None
    )


def _tree_transform(adjacency, source, destination):
  queue = [(source, np.eye(4))]
  visited = {source}
  while queue:
    current, source_to_current = queue.pop(0)
    if current == destination:
      return source_to_current
    for neighbor, current_to_neighbor in adjacency.get(current, []):
      if neighbor in visited:
        continue
      visited.add(neighbor)
      queue.append((
        neighbor, current_to_neighbor @ source_to_current
      ))
  return None


def export_extrinsic_quality(calib, camera_names, pose_table=None):
  """Export non-invasive per-camera and per-pair quality diagnostics."""
  valid = np.asarray(calib.valid, dtype=bool)
  inliers = np.asarray(calib.inliers, dtype=bool)
  errors, _ = tables.reprojection_error(
    calib.reprojected, calib.point_table
  )
  errors = np.asarray(errors, dtype=np.float64)
  point_valid = np.asarray(calib.point_table.valid, dtype=bool)
  final_camera_pose_table = calib.camera_poses.pose_table
  final_camera_poses = np.asarray(
    final_camera_pose_table.poses, dtype=np.float64
  ).reshape(-1, 4, 4)
  final_camera_valid = np.asarray(
    final_camera_pose_table.valid, dtype=bool
  ).reshape(-1)
  camera_stats = {}
  for camera_index, camera_name in enumerate(camera_names):
    camera_valid = valid[camera_index]
    camera_inliers = inliers[camera_index]
    observation_count = int(np.count_nonzero(camera_valid))
    inlier_count = int(np.count_nonzero(camera_inliers))
    frame_reduction_axes = tuple(range(1, camera_valid.ndim))
    valid_frames = np.any(
      camera_valid, axis=frame_reduction_axes
    )
    inlier_frames = np.any(
      camera_inliers, axis=frame_reduction_axes
    )
    camera_stats[camera_name] = struct(
      observation_count=observation_count,
      inlier_count=inlier_count,
      outlier_count=observation_count - inlier_count,
      inlier_ratio=(
        float(inlier_count / observation_count)
        if observation_count else None
      ),
      detected_frame_count=int(np.count_nonzero(valid_frames)),
      inlier_frame_count=int(np.count_nonzero(inlier_frames)),
      rejected_frame_count=int(np.count_nonzero(
        valid_frames & ~inlier_frames
      )),
      reprojection_RMS_px=_masked_rms(
        errors[camera_index], camera_inliers
      ),
      all_points_RMS_px=_masked_rms(
        errors[camera_index], camera_valid
      )
    )

  selected_edges = set()
  selected_master = None
  if pose_table is not None:
    overlaps = tables.pattern_overlaps(pose_table, axis=0)
    selected_master, pairs = graph.select_pairs(
      overlaps.copy(), hop_penalty=0.9
    )
    selected_edges = {
      tuple(sorted((int(first), int(second))))
      for first, second in pairs
    }

  pair_stats = {}
  pair_transforms = {}
  warnings = []
  for camera_name, statistics in camera_stats.items():
    if (
        statistics.inlier_ratio is not None and
        statistics.inlier_ratio < 0.9):
      warnings.append(struct(
        code="camera_high_outlier_ratio",
        camera=camera_name,
        message=(
          "{} retained {:.2%} of observations and rejected {} frames"
        ).format(
          camera_name,
          statistics.inlier_ratio,
          statistics.rejected_frame_count
        )
      ))
  for first in range(len(camera_names)):
    for second in range(first + 1, len(camera_names)):
      name_first = camera_names[first]
      name_second = camera_names[second]
      pair_name = "{}:{}".format(name_first, name_second)
      shared_points = point_valid[first] & point_valid[second]
      shared_inliers = inliers[first] & inliers[second]
      shared_frame_board = np.any(
        shared_points, axis=-1
      )
      shared_frames = np.any(
        shared_frame_board, axis=1
      )
      final_transform = None
      if final_camera_valid[first] and final_camera_valid[second]:
        final_transform = (
          final_camera_poses[second]
          @ np.linalg.inv(final_camera_poses[first])
        )
      pose_stats = _pair_pose_statistics(
        pose_table, first, second,
        final_transform=final_transform
      )
      if pose_stats.transform is not None:
        pair_transforms[(first, second)] = pose_stats.transform
      rotation_scatter = pose_stats.rotation_scatter_deg
      translation_scatter = pose_stats.translation_scatter
      status = _pair_quality_status(
        int(np.count_nonzero(shared_frames)),
        rotation_scatter,
        translation_scatter
      )
      pair_errors = np.concatenate([
        errors[first][shared_inliers],
        errors[second][shared_inliers]
      ])
      pair_stats[pair_name] = struct(
        cameras=[name_first, name_second],
        selected_initialization_edge=(
          (first, second) in selected_edges
        ),
        common_frame_count=int(np.count_nonzero(shared_frames)),
        common_corner_count=int(np.count_nonzero(shared_points)),
        common_inlier_corner_count=int(np.count_nonzero(shared_inliers)),
        pair_reprojection_RMS_px=_rms(pair_errors),
        common_pose_count=pose_stats.common_pose_count,
        pose_inlier_count=pose_stats.pose_inlier_count,
        rotation_scatter_deg=rotation_scatter,
        translation_scatter=translation_scatter,
        frobenius_scatter=pose_stats.frobenius_scatter,
        final_edge_rotation_residual_deg=(
          pose_stats.final_edge_rotation_residual_deg
        ),
        final_edge_translation_residual=(
          pose_stats.final_edge_translation_residual
        ),
        final_rotation_residual_rms_deg=(
          pose_stats.final_rotation_residual_rms_deg
        ),
        final_translation_residual_rms=(
          pose_stats.final_translation_residual_rms
        ),
        status=status
      )
      if status == "weak":
        warnings.append(struct(
          code="weak_extrinsic_link",
          pair=pair_name,
          message=(
            "{} is a weak extrinsic link: {} common frames, "
            "rotation scatter {}, translation scatter {}".format(
              pair_name,
              int(np.count_nonzero(shared_frames)),
              (
                "{:.4f} deg".format(rotation_scatter)
                if rotation_scatter is not None else "unavailable"
              ),
              (
                "{:.4f}".format(translation_scatter)
                if translation_scatter is not None else "unavailable"
              )
            )
          )
        ))

  tree_adjacency = {index: [] for index in range(len(camera_names))}
  for first, second in selected_edges:
    transform = pair_transforms.get((first, second))
    if transform is None:
      continue
    tree_adjacency[first].append((second, transform))
    tree_adjacency[second].append((first, np.linalg.inv(transform)))
  redundant_consistency = {}
  for (first, second), direct_transform in pair_transforms.items():
    if (first, second) in selected_edges:
      continue
    tree_transform = _tree_transform(
      tree_adjacency, first, second
    )
    if tree_transform is None:
      continue
    closure = matrix.pose_errors(
      tree_transform.reshape(1, 4, 4),
      direct_transform.reshape(1, 4, 4)
    )
    pair_name = "{}:{}".format(
      camera_names[first], camera_names[second]
    )
    redundant_consistency[pair_name] = struct(
      rotation_closure_error_deg=float(closure.rotation_deg[0]),
      translation_closure_error=float(closure.translation[0]),
      frobenius_closure_error=float(closure.frobius[0])
    )

  selected_edge_names = [
    "{}:{}".format(camera_names[first], camera_names[second])
    for first, second in sorted(selected_edges)
  ]
  return struct(
    diagnostic_only=True,
    translation_unit=(
      "same length unit as calibration board dimensions"
    ),
    thresholds=EXTRINSIC_QUALITY_THRESHOLDS,
    cameras=camera_stats,
    pairs=pair_stats,
    graph=struct(
      connected=(
        len(selected_edges) == max(0, len(camera_names) - 1)
      ),
      initialization_master=(
        camera_names[selected_master]
        if selected_master is not None else None
      ),
      selected_initialization_edges=selected_edge_names,
      weak_selected_edges=[
        pair_name for pair_name in selected_edge_names
        if pair_stats[pair_name].status == "weak"
      ],
      redundant_edge_consistency=redundant_consistency
    ),
    warnings=warnings
  )


def export_intrinsic_quality(camera, rms, input_image_count=None):
  """Export the OpenCV intrinsic-calibration result for one camera."""
  per_view = np.asarray(
    camera.error_perview
    if camera.error_perview is not None else [],
    dtype=np.float64
  ).reshape(-1)
  dataset = camera.intrinsic_dataset or {}
  image_ids = [int(value) for value in dataset.get("image_ids", [])]
  board_ids = [int(value) for value in dataset.get("board_ids", [])]
  point_counts = [
    int(value) for value in dataset.get("point_counts", [])
  ]
  view_count = int(per_view.size)

  views = []
  if (
      len(image_ids) == view_count and
      len(board_ids) == view_count and
      len(point_counts) == view_count):
    views = [
      struct(
        image_index=image_id,
        board_index=board_id,
        observation_count=point_count,
        RMS=float(view_rms)
      )
      for image_id, board_id, point_count, view_rms in zip(
        image_ids, board_ids, point_counts, per_view
      )
    ]

  detected_view_count = int(
    dataset.get("detected_view_count", view_count)
  )
  candidate_view_count = int(
    dataset.get("candidate_view_count", detected_view_count)
  )
  used_image_count = len(set(image_ids)) if image_ids else view_count
  detected_image_count = int(
    dataset.get("detected_image_count", used_image_count)
  )
  candidate_image_count = int(
    dataset.get("candidate_image_count", detected_image_count)
  )
  quality_candidate_image_count = int(
    dataset.get("quality_candidate_image_count", detected_image_count)
  )
  input_image_count = (
    int(input_image_count)
    if input_image_count is not None else detected_image_count
  )
  return struct(
    RMS=float(rms),
    mean_view_RMS=(
      float(np.mean(per_view)) if view_count else None
    ),
    max_view_RMS=(
      float(np.max(per_view)) if view_count else None
    ),
    unit="px",
    view_count=view_count,
    image_count=used_image_count,
    input_image_count=input_image_count,
    detected_image_count=detected_image_count,
    candidate_image_count=candidate_image_count,
    detection_failed_image_count=max(
      input_image_count - detected_image_count, 0
    ),
    quality_rejected_image_count=max(
      detected_image_count - quality_candidate_image_count, 0
    ),
    excluded_by_limit_image_count=max(
      quality_candidate_image_count - candidate_image_count, 0
    ),
    rejected_image_count=max(
      candidate_image_count - used_image_count, 0
    ),
    observation_count=(
      int(sum(point_counts))
      if len(point_counts) == view_count else None
    ),
    detected_view_count=detected_view_count,
    quality_candidate_view_count=int(
      dataset.get("quality_candidate_view_count", detected_view_count)
    ),
    candidate_view_count=candidate_view_count,
    rejected_view_count=max(candidate_view_count - view_count, 0),
    min_board_coverage=dataset.get("min_board_coverage"),
    view_error_limit=dataset.get("view_error_limit"),
    view_mad_scale=dataset.get("view_mad_scale"),
    filter_iterations=dataset.get("filter_iterations"),
    filter_converged=dataset.get("filter_converged"),
    filter_limit_reached=dataset.get("filter_limit_reached"),
    selection_seed=dataset.get("selection_seed"),
    filter_history=dataset.get("filter_history", []),
    reprojection_rejected_image_ids=dataset.get(
      "reprojection_rejected_image_ids", []
    ),
    overall_error_limit=dataset.get("overall_error_limit"),
    overall_error_limit_met=dataset.get("overall_error_limit_met"),
    quality_rejected_image_ids=dataset.get(
      "quality_rejected_image_ids", []
    ),
    image_board_coverage=dataset.get("image_board_coverage", {}),
    views=views
  )


def export_single(
    filename, cameras, camera_names, filenames, errors=None):
  camera_filenames = filenames
  filenames = transpose_lists(filenames)
  qualities = (
    [
      export_intrinsic_quality(camera, error, len(camera_files))
      for camera, error, camera_files in zip(
        cameras, errors, camera_filenames
      )
    ]
    if errors is not None else None
  )
  data = struct(
    cameras = export_cameras(camera_names, cameras, qualities),
    image_sets = export_images(camera_names, filenames)
  )
 
  with open(filename, 'w') as outfile:
    json.dump(to_dicts(data), outfile, indent=2)

def export(filename, calib, names, filenames, master=None):  
  data = export_json(calib, names, filenames, master=master)
  
  with open(filename, 'w') as outfile:
    json.dump(to_dicts(data), outfile, indent=2)


def export_json(
    calib, names, filenames, master=None, pose_table=None):
  if master is not None:
    calib = calib.with_master(master)

  camera_poses = calib.camera_poses.pose_table
  filenames = transpose_lists(filenames)

  data = struct(
    cameras = export_cameras(names.camera, calib.cameras),
    # camera_poses = export_sequential(names.camera, camera_poses),
    camera_poses = export_camera_poses(names.camera, camera_poses)\
      if master is None else export_relative(names.camera, camera_poses, master),
    image_sets = export_images(names.camera, filenames),
    quality = export_calibration_quality(calib),
    extrinsic_quality = export_extrinsic_quality(
      calib, names.camera, pose_table=pose_table
    )

  )

  return data
  
