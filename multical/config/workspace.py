from structs.struct import map_none
from multical.io.import_calib import load_calibration
from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.workspace import Workspace

from .arguments import *


def get_motion_model(motion_model):
    if motion_model == "rolling":
        return RollingFrames
    elif motion_model == "static":
        return StaticFrames
    else:
        assert False, f"unknown motion model {motion_model}, (static|rolling)"


def initialise_with_images(ws : Workspace, boards, camera_images, 
  camera_opts : CameraOpts = CameraOpts(), runtime : RuntimeOpts = RuntimeOpts()):


    ws.add_camera_images(camera_images, j=runtime.num_threads)
    ws.detect_boards(boards, load_cache=not runtime.no_cache, j=runtime.num_threads)

    calib = map_none(load_calibration, camera_opts.calibration)

    if calib is not None:
      ws.set_calibration(calib.cameras)
    else:
      ws.calibrate_single(camera_opts.distortion_model, camera_opts.intrinsic_error_limit,
          fix_aspect=camera_opts.fix_aspect,
          has_skew=camera_opts.allow_skew, 
          max_images=camera_opts.limit_intrinsic,
          min_board_coverage=camera_opts.intrinsic_min_board_coverage,
          view_error_limit=camera_opts.intrinsic_view_error_limit,
          view_mad_scale=camera_opts.intrinsic_view_mad_scale,
          filter_iterations=camera_opts.intrinsic_filter_iterations,
          max_reject_fraction=camera_opts.intrinsic_max_reject_fraction,
          min_views=camera_opts.intrinsic_min_views,
          selection_seed=runtime.seed,
          isFisheye=camera_opts.isFisheye)

    ws.initialise_poses(
        motion_model=get_motion_model(camera_opts.motion_model),
        camera_poses=calib.camera_poses if calib is not None else None,
        exclude_bad_poses=runtime.exclude_bad_poses,
        pose_error_limit=runtime.pose_error_limit,
        is_non_overlapping=runtime.is_non_overlapping
      )
    return ws


def optimize(ws : Workspace, opt : OptimizerOpts = OptimizerOpts()):

  ws.calibrate("calibration", loss=opt.loss,
    boards=opt.adjust_board,
    cameras=not opt.fix_intrinsic,
    camera_poses=not opt.fix_camera_poses,
    board_poses=not opt.fix_board_poses,
    motion=not opt.fix_motion,
    num_adjustments=opt.iter,
    initial_loss=opt.initial_loss,
    auto_scale=opt.auto_scale, 
    outlier_threshold=opt.outlier_threshold,
    outlier_min_threshold=opt.outlier_min_threshold,
    outlier_max_threshold=opt.outlier_max_threshold,
    frame_outlier_ratio=opt.frame_outlier_ratio,
    frame_outlier_min_points=opt.frame_outlier_min_points,
    final_recheck_iterations=opt.final_recheck_iterations,
    quantile=opt.outlier_quantile)

  return ws
