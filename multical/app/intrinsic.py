from multical.io.export_calib import export_single
from multical.io.distortion_check import export_distortion_checks
from multical.camera import calibrate_cameras
from multical.workspace import detect_boards_cached
from os import path
import pathlib
from multical.config.runtime import find_board_config, find_camera_images
from multical.image.detect import common_image_size
from multical.io.logging import setup_logging
from multical.io.logging import info
import json

from structs.struct import  map_list, pformat_struct, split_dict
from multical import image

from structs.numpy import struct, shape

from multical.config.arguments import *

@dataclass
class Intrinsic:
  """Run separate intrinsic calibration for set of cameras"""
  paths     : PathOpts = PathOpts(name="intrinsic")
  camera    : CameraOpts = CameraOpts()
  runtime   : RuntimeOpts = RuntimeOpts()

  def execute(self):
      calibrate_intrinsic(self)


def setup_paths(paths):
  output_path = paths.output_path or paths.image_path
  temp_folder = pathlib.Path(output_path).joinpath("." + paths.name)
  temp_folder.mkdir(exist_ok=True, parents=True)

  return struct(
    output = output_path,
    temp=str(temp_folder),

    calibration_file=path.join(output_path, f"{paths.name}.json"),
    log_file=str(temp_folder.joinpath("log.txt")),
    detections=str(temp_folder.joinpath("detections.pkl"))
  )


def calibrate_intrinsic(args):
    paths=setup_paths(args.paths)

    setup_logging(args.runtime.log_level, [], log_file=paths.log_file)
    info(pformat_struct(args)) 

    image_path = os.path.expanduser(args.paths.image_path)
    info(f"Finding images in {image_path}")

    camera_images = find_camera_images(image_path, 
      args.paths.cameras, args.paths.camera_pattern, matching=False)

    image_counts = {k:len(files) for k, files in zip(camera_images.cameras, camera_images.filenames)}
    info("Found camera directories with images {}".format(image_counts))

    board_names, boards = split_dict(find_board_config(image_path, args.paths.boards))

    info("Loading images..")
    images = image.detect.load_images(camera_images.filenames,  
      prefix=camera_images.image_path, j=args.runtime.num_threads)
    image_sizes = map_list(common_image_size, images)


    info({k:image_size for k, image_size in zip(camera_images.cameras, image_sizes)})
    cache_key = struct(boards=boards, image_sizes=image_sizes, filenames=camera_images.filenames)

    detected_points = detect_boards_cached(boards, images, 
        paths.detections, cache_key, j=args.runtime.num_threads)

    cameras, errs = calibrate_cameras(
      boards, detected_points, image_sizes,
      args.camera.intrinsic_error_limit,
      model=args.camera.distortion_model,
      fix_aspect=args.camera.fix_aspect,
      max_images=args.camera.limit_intrinsic,
      min_board_coverage=args.camera.intrinsic_min_board_coverage,
      view_error_limit=args.camera.intrinsic_view_error_limit,
      view_mad_scale=args.camera.intrinsic_view_mad_scale,
      filter_iterations=args.camera.intrinsic_filter_iterations,
      max_reject_fraction=args.camera.intrinsic_max_reject_fraction,
      min_views=args.camera.intrinsic_min_views,
      selection_seed=args.runtime.seed)
     
    for name, camera, err in zip(camera_images.cameras, cameras, errs):
        info(f"Calibrated {name}, with RMS={err:.2f}")
        info(camera)
        info("")

    filter_report = {
      "mode": "robust_view_filter",
      "min_board_coverage": (
        args.camera.intrinsic_min_board_coverage
      ),
      "view_error_limit": args.camera.intrinsic_view_error_limit,
      "view_mad_scale": args.camera.intrinsic_view_mad_scale,
      "filter_iterations": args.camera.intrinsic_filter_iterations,
      "max_reject_fraction": (
        args.camera.intrinsic_max_reject_fraction
      ),
      "min_views": args.camera.intrinsic_min_views,
      "selection_seed": args.runtime.seed,
      "cameras": {}
    }
    for name, camera, filenames in zip(
        camera_images.cameras, cameras, camera_images.filenames):
      dataset = camera.intrinsic_dataset

      def image_names(indices):
        return [
          filenames[int(index)] for index in indices
          if 0 <= int(index) < len(filenames)
        ]

      used_ids = sorted(set(
        int(value) for value in dataset.get("image_ids", [])
      ))
      detected_ids = dataset.get("detected_image_ids", [])
      candidate_ids = dataset.get("candidate_image_ids", [])
      filter_report["cameras"][name] = {
        "input_image_count": len(filenames),
        "used_images": image_names(used_ids),
        "detection_failed_images": image_names(
          sorted(set(range(len(filenames))) - set(detected_ids))
        ),
        "incomplete_board_images": image_names(
          dataset.get("quality_rejected_image_ids", [])
        ),
        "excluded_by_limit_images": image_names(
          sorted(
            set(dataset.get("quality_candidate_image_ids", [])) -
            set(candidate_ids)
          )
        ),
        "reprojection_rejected_images": image_names(
          sorted(set(candidate_ids) - set(used_ids))
        ),
        "reprojection_filter_history": dataset.get(
          "filter_history", []
        ),
        "reprojection_filter_converged": dataset.get(
          "filter_converged"
        ),
        "reprojection_filter_limit_reached": dataset.get(
          "filter_limit_reached"
        ),
        "image_board_coverage": {
          filenames[int(index)]: value
          for index, value in dataset.get(
            "image_board_coverage", {}
          ).items()
          if 0 <= int(index) < len(filenames)
        }
      }
    filter_report_path = pathlib.Path(paths.output) / (
      f"{args.paths.name}_filter.json"
    )
    filter_report_path.write_text(
      json.dumps(filter_report, indent=2) + "\n",
      encoding="utf-8"
    )
    info("Wrote intrinsic image filtering report to {}".format(
      filter_report_path
    ))

    info(f"Writing single calibrations to {paths.calibration_file}")
    export_single(
      paths.calibration_file,
      cameras,
      camera_images.cameras,
      camera_images.filenames,
      errors=errs
    )
    _, distortion_directory = export_distortion_checks(
      paths.output,
      camera_images.cameras,
      cameras,
      images,
      camera_images.filenames,
      stage="intrinsic"
    )
    info("Wrote distortion validation images to {}".format(
      distortion_directory
    ))


if __name__ == '__main__':
  run_with(Intrinsic)
