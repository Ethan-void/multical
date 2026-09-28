"""Join manual pixel clicks with named measured world coordinates."""

import argparse
from pathlib import Path

import numpy as np
import yaml


def _load_mapping(filename, description):
  path = Path(filename).expanduser().resolve()
  data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  if not isinstance(data, dict):
    raise ValueError("{} must be a YAML mapping".format(description))
  return path, data


def join_world_observations(observe_file, world_points_file, output_file):
  observe_path, observe = _load_mapping(observe_file, "observe file")
  points_path, point_data = _load_mapping(
    world_points_file, "world points file"
  )
  points = point_data.get("points")
  frames = observe.get("frames")
  if not isinstance(points, dict) or not points:
    raise ValueError("world points file must contain a non-empty points map")
  if not isinstance(frames, list) or not frames:
    raise ValueError("observe file must contain a non-empty frames list")

  observations = []
  used_points = set()
  for index, frame in enumerate(frames):
    if not isinstance(frame, dict) or frame.get("frame") is None:
      raise ValueError("observe frame {} is invalid".format(index))
    name = str(frame.get("point_name", frame["frame"]))
    if name not in points:
      continue
    world_point = np.asarray(points[name], dtype=np.float64)
    if world_point.shape != (3,) or not np.isfinite(world_point).all():
      raise ValueError("world point {} must be finite [X, Y, Z]".format(name))
    pixels = frame.get("observations")
    if not isinstance(pixels, dict) or len(pixels) < 2:
      continue
    for camera, pixel in pixels.items():
      image_point = np.asarray(pixel, dtype=np.float64)
      if image_point.shape != (2,) or not np.isfinite(image_point).all():
        raise ValueError("{} pixel for {} must be finite [u, v]".format(
          name, camera
        ))
      observations.append({
        "capture": name,
        "camera": str(camera),
        "world_point": [float(value) for value in world_point],
        "image_point": [float(value) for value in image_point],
        "source_frame": str(frame.get("source_frame", frame["frame"]))
      })
    used_points.add(name)

  if len(used_points) < 4:
    raise ValueError(
      "fewer than four named points have both world coordinates and "
      "multi-camera pixel clicks"
    )
  output = {
    "coordinate_frame": "world",
    "world_units": str(point_data.get("world_units", "meters")),
    "source": {
      "observe": str(observe_path),
      "world_points": str(points_path),
      "image_path": (observe.get("source") or {}).get("image_path"),
      "mode": "manual_click_joined"
    },
    "cameras": [str(value) for value in observe.get("cameras", [])],
    "observations": observations
  }
  destination = Path(output_file).expanduser().resolve()
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
    encoding="utf-8"
  )
  print("Joined {} points / {} observations into {}".format(
    len(used_points), len(observations), destination
  ))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--observe", required=True)
  parser.add_argument("--world_points", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  join_world_observations(args.observe, args.world_points, args.output)


if __name__ == "__main__":
  main()
