"""Interactively annotate synchronized multi-camera pixel observations."""

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import yaml
from simple_parsing.helpers import list_field

from multical.config.arguments import run_with
from multical.config.runtime import find_camera_images


HEADER_HEIGHT = 32
FOOTER_HEIGHT = 54
MAGNIFIER_SIZE = 180
MAGNIFIER_RADIUS_PX = 24
SIDEBAR_WIDTH = 190
SIDEBAR_HEADER_HEIGHT = 44
SIDEBAR_ROW_HEIGHT = 30


def compose_mosaic(
    camera_names, camera_images, observations,
    columns=2, tile_width=640):
  """Build a camera mosaic and return raw-pixel mapping metadata."""
  if columns < 1 or tile_width < 100:
    raise ValueError("columns must be >= 1 and tile_width must be >= 100")
  if len(camera_names) != len(camera_images):
    raise ValueError("camera names and images must have equal length")
  image_height = int(round(tile_width * 0.75))
  rows = max(1, int(np.ceil(len(camera_names) / float(columns))))
  cell_height = HEADER_HEIGHT + image_height
  canvas = np.full(
    (rows * cell_height + FOOTER_HEIGHT, columns * tile_width, 3),
    35,
    dtype=np.uint8
  )
  placements = []

  for index, (camera_name, source) in enumerate(
      zip(camera_names, camera_images)):
    if source is None:
      raise ValueError("could not load image for {}".format(camera_name))
    source_bgr = (
      cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
      if source.ndim == 2 else source
    )
    row, column = divmod(index, columns)
    cell_x = column * tile_width
    cell_y = row * cell_height
    height, width = source_bgr.shape[:2]
    scale = min(
      float(tile_width) / float(width),
      float(image_height) / float(height)
    )
    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))
    image_x = cell_x + (tile_width - display_width) // 2
    image_y = (
      cell_y + HEADER_HEIGHT +
      (image_height - display_height) // 2
    )
    resized = cv2.resize(
      source_bgr,
      (display_width, display_height),
      interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    canvas[
      image_y:image_y + display_height,
      image_x:image_x + display_width
    ] = resized
    cv2.putText(
      canvas, str(camera_name), (cell_x + 10, cell_y + 23),
      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
      1, cv2.LINE_AA
    )
    placement = {
      "camera": str(camera_name),
      "x": image_x,
      "y": image_y,
      "width": display_width,
      "height": display_height,
      "scale": scale,
      "source_width": width,
      "source_height": height
    }
    placements.append(placement)

    if camera_name in observations:
      raw_x, raw_y = observations[camera_name]
      display_x = int(round(image_x + raw_x * scale))
      display_y = int(round(image_y + raw_y * scale))
      cv2.drawMarker(
        canvas, (display_x, display_y), (0, 0, 255),
        cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA
      )
      cv2.circle(
        canvas, (display_x, display_y), 3,
        (0, 255, 255), 1, cv2.LINE_AA
      )
      cv2.putText(
        canvas,
        "({:.1f}, {:.1f})".format(raw_x, raw_y),
        (min(display_x + 9, cell_x + tile_width - 170),
         max(display_y - 9, cell_y + HEADER_HEIGHT + 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 255, 255), 1, cv2.LINE_AA
      )

      raw_x0 = max(0, int(np.floor(raw_x - MAGNIFIER_RADIUS_PX)))
      raw_y0 = max(0, int(np.floor(raw_y - MAGNIFIER_RADIUS_PX)))
      raw_x1 = min(
        width, int(np.ceil(raw_x + MAGNIFIER_RADIUS_PX + 1))
      )
      raw_y1 = min(
        height, int(np.ceil(raw_y + MAGNIFIER_RADIUS_PX + 1))
      )
      patch = source_bgr[raw_y0:raw_y1, raw_x0:raw_x1]
      magnifier_size = min(
        MAGNIFIER_SIZE,
        max(80, display_width // 3),
        max(80, display_height // 2)
      )
      magnifier_x = image_x + display_width - magnifier_size - 6
      magnifier_y = image_y + 6
      magnified = cv2.resize(
        patch,
        (magnifier_size, magnifier_size),
        interpolation=cv2.INTER_NEAREST
      )
      canvas[
        magnifier_y:magnifier_y + magnifier_size,
        magnifier_x:magnifier_x + magnifier_size
      ] = magnified
      cv2.rectangle(
        canvas,
        (magnifier_x, magnifier_y),
        (
          magnifier_x + magnifier_size - 1,
          magnifier_y + magnifier_size - 1
        ),
        (0, 255, 255), 2
      )
      magnified_x = int(round(
        magnifier_x +
        (raw_x - raw_x0) * magnifier_size / (raw_x1 - raw_x0)
      ))
      magnified_y = int(round(
        magnifier_y +
        (raw_y - raw_y0) * magnifier_size / (raw_y1 - raw_y0)
      ))
      cv2.drawMarker(
        canvas, (magnified_x, magnified_y), (0, 0, 255),
        cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA
      )
      placement["magnifier"] = {
        "x": magnifier_x,
        "y": magnifier_y,
        "size": magnifier_size,
        "raw_x": raw_x0,
        "raw_y": raw_y0,
        "raw_width": raw_x1 - raw_x0,
        "raw_height": raw_y1 - raw_y0
      }

  return canvas, placements


def compose_point_sidebar(
    canvas, point_names, annotations, current_index,
    width=SIDEBAR_WIDTH):
  """Append a clickable navigator for saved point observations."""
  height = canvas.shape[0]
  sidebar = np.full((height, width, 3), 28, dtype=np.uint8)
  cv2.putText(
    sidebar, "Saved points", (12, 28),
    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255),
    1, cv2.LINE_AA
  )
  cv2.line(
    sidebar, (0, SIDEBAR_HEADER_HEIGHT - 1),
    (width - 1, SIDEBAR_HEADER_HEIGHT - 1),
    (90, 90, 90), 1, cv2.LINE_AA
  )
  visible_rows = max(
    1, (height - SIDEBAR_HEADER_HEIGHT - 8) // SIDEBAR_ROW_HEIGHT
  )
  start = max(
    0,
    min(
      current_index - visible_rows // 2,
      max(0, len(point_names) - visible_rows)
    )
  )
  end = min(len(point_names), start + visible_rows)
  hitboxes = []
  for visible_index, point_index in enumerate(range(start, end)):
    name = str(point_names[point_index])
    saved = name in annotations
    active = point_index == current_index
    top = SIDEBAR_HEADER_HEIGHT + visible_index * SIDEBAR_ROW_HEIGHT
    bottom = top + SIDEBAR_ROW_HEIGHT - 2
    if active:
      color = (135, 85, 25)
    elif saved:
      color = (48, 105, 48)
    else:
      color = (48, 48, 48)
    cv2.rectangle(
      sidebar, (6, top + 2), (width - 7, bottom),
      color, -1, cv2.LINE_AA
    )
    status = "saved" if saved else "new"
    cv2.putText(
      sidebar, "{}  {}".format(name, status),
      (14, top + 21),
      cv2.FONT_HERSHEY_SIMPLEX, 0.52,
      (255, 255, 255) if saved or active else (160, 160, 160),
      1, cv2.LINE_AA
    )
    if saved:
      hitboxes.append({
        "x0": canvas.shape[1] + 6,
        "x1": canvas.shape[1] + width - 7,
        "y0": top + 2,
        "y1": bottom,
        "index": point_index,
        "point_name": name
      })
  combined = np.concatenate([canvas, sidebar], axis=1)
  return combined, hitboxes


def sidebar_hit_test(hitboxes, x, y):
  for hitbox in hitboxes:
    if (
        hitbox["x0"] <= x <= hitbox["x1"] and
        hitbox["y0"] <= y <= hitbox["y1"]):
      return hitbox["index"]
  return None


def map_mosaic_click(placements, x, y):
  """Map a mosaic click back to one camera's original pixel coordinates."""
  for placement in placements:
    magnifier = placement.get("magnifier")
    if magnifier is None:
      continue
    if (
        magnifier["x"] <= x < magnifier["x"] + magnifier["size"] and
        magnifier["y"] <= y < magnifier["y"] + magnifier["size"]):
      raw_x = (
        magnifier["raw_x"] +
        (float(x) - magnifier["x"]) *
        magnifier["raw_width"] / magnifier["size"]
      )
      raw_y = (
        magnifier["raw_y"] +
        (float(y) - magnifier["y"]) *
        magnifier["raw_height"] / magnifier["size"]
      )
      return placement["camera"], [raw_x, raw_y]

  for placement in placements:
    if (
        placement["x"] <= x < placement["x"] + placement["width"] and
        placement["y"] <= y < placement["y"] + placement["height"]):
      raw_x = (float(x) - placement["x"]) / placement["scale"]
      raw_y = (float(y) - placement["y"]) / placement["scale"]
      raw_x = float(np.clip(
        raw_x, 0.0, placement["source_width"] - 1.0
      ))
      raw_y = float(np.clip(
        raw_y, 0.0, placement["source_height"] - 1.0
      ))
      return placement["camera"], [raw_x, raw_y]
  return None


def nudge_observation(
    observations, active_camera, placements, delta_x, delta_y):
  """Move one selected raw-image point while keeping it inside the image."""
  if active_camera not in observations:
    return False
  placement = next(
    (
      value for value in placements
      if value["camera"] == active_camera
    ),
    None
  )
  if placement is None:
    return False
  point = observations[active_camera]
  point[0] = float(np.clip(
    point[0] + delta_x, 0.0, placement["source_width"] - 1.0
  ))
  point[1] = float(np.clip(
    point[1] + delta_y, 0.0, placement["source_height"] - 1.0
  ))
  return True


def load_existing_observations(filename):
  path = Path(filename)
  if not path.is_file():
    return {}
  data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  frames = data.get("frames", []) if isinstance(data, dict) else []
  return {
    str(frame["frame"]): {
      str(camera): [float(point[0]), float(point[1])]
      for camera, point in frame.get("observations", {}).items()
    }
    for frame in frames
    if isinstance(frame, dict) and frame.get("frame") is not None
  }


def observation_work_items(frame_names, point_names=None):
  """Map source image frames to unique observation identifiers."""
  point_names = list(point_names or [])
  if not point_names:
    return [
      {"frame": str(frame), "source_frame": str(frame)}
      for frame in frame_names
    ]
  normalized = [str(name).strip() for name in point_names]
  if any(not name for name in normalized):
    raise ValueError("point names cannot be empty")
  if len(set(normalized)) != len(normalized):
    raise ValueError("point names must be unique")
  single_source = len(frame_names) == 1
  return [
    {
      "frame": (
        point_name if single_source else
        "{}:{}".format(source_frame, point_name)
      ),
      "source_frame": str(source_frame),
      "point_name": point_name
    }
    for source_frame in frame_names
    for point_name in normalized
  ]


def automatic_point_work_items(source_frame, annotations):
  """Build saved Pxx items plus one new item for continuous annotation."""
  numbered = []
  for name in annotations:
    match = re.fullmatch(r"P([0-9]+)", str(name), flags=re.IGNORECASE)
    if match:
      numbered.append(int(match.group(1)))
  next_number = max(numbered, default=0) + 1
  return [
    {
      "frame": "P{:02d}".format(number),
      "source_frame": str(source_frame),
      "point_name": "P{:02d}".format(number)
    }
    for number in range(1, next_number + 1)
  ], next_number - 1


def append_automatic_point(work_items):
  """Append the next sequential Pxx work item."""
  last_number = int(work_items[-1]["point_name"][1:])
  next_number = last_number + 1
  point_name = "P{:02d}".format(next_number)
  work_items.append({
    "frame": point_name,
    "source_frame": work_items[-1]["source_frame"],
    "point_name": point_name
  })


def write_observations(
    filename, image_path, cameras, frame_names, annotations,
    minimum_cameras=2, source_frames=None):
  path = Path(filename).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  frames = []
  for frame in frame_names:
    if (
        frame not in annotations or
        len(annotations[frame]) < minimum_cameras):
      continue
    record = {
      "frame": frame,
      "observations": {
        camera: [
          round(float(point[0]), 3),
          round(float(point[1]), 3)
        ]
        for camera, point in annotations[frame].items()
      }
    }
    if source_frames and source_frames.get(frame) != frame:
      record["source_frame"] = source_frames[frame]
    frames.append(record)
  output = {
    "sequence": Path(image_path).resolve().name,
    "source": {
      "image_path": str(Path(image_path).resolve()),
      "mode": "manual_click"
    },
    "cameras": list(cameras),
    "frames": frames
  }
  path.write_text(
    yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
    encoding="utf-8"
  )
  return output, path


def annotate_observations(
    image_path, cameras, output, camera_pattern=None,
    columns=2, tile_width=640, start_frame=None, frame=None,
    points=None):
  input_path = Path(image_path).expanduser().resolve()
  if not input_path.exists():
    raise ValueError(
      "image_path does not exist: {}. Use the actual image path; "
      "for example /Users/name/Pictures/test.jpg".format(input_path)
    )
  if input_path.is_file():
    if len(cameras) != 1:
      raise ValueError(
        "single-image mode requires exactly one camera name"
      )
    camera_images = SimpleNamespace(
      image_path=str(input_path.parent),
      cameras=list(cameras),
      image_names=[input_path.name],
      filenames=[[input_path.name]]
    )
  else:
    if not input_path.is_dir():
      raise ValueError(
        "image_path must be an image file or dataset directory: {}".format(
          input_path
        )
      )
    camera_images = find_camera_images(
      image_path, cameras, camera_pattern, matching=True
    )
  all_frame_names = list(camera_images.image_names)
  if not all_frame_names:
    raise ValueError("no synchronized image filenames were found")
  if frame is not None:
    if frame not in all_frame_names:
      raise ValueError(
        "requested frame {} was not found in every camera".format(frame)
      )
    frame_names = [frame]
  else:
    frame_names = all_frame_names
  annotations = load_existing_observations(output)
  automatic_points = frame is not None and not list(points or [])
  if automatic_points:
    work_items, index = automatic_point_work_items(
      frame_names[0], annotations
    )
  else:
    work_items = observation_work_items(frame_names, points)
    index = 0
  observation_names = [item["frame"] for item in work_items]
  source_frames = {
    item["frame"]: item["source_frame"] for item in work_items
  }
  minimum_cameras = min(2, len(camera_images.cameras))
  if start_frame is not None:
    matching = [
      item_index for item_index, item in enumerate(work_items)
      if (
        item["frame"] == start_frame or
        item["source_frame"] == start_frame
      )
    ]
    if matching:
      index = matching[0]
  window_name = "Multical synchronized pixel observations"
  state = {
    "placements": [],
    "sidebar_hitboxes": [],
    "observations": {},
    "active_camera": None,
    "jump_index": None
  }

  def mouse_callback(event, x, y, _flags, _parameter):
    if event == cv2.EVENT_LBUTTONDOWN:
      jump_index = sidebar_hit_test(
        state["sidebar_hitboxes"], x, y
      )
      if jump_index is not None:
        state["jump_index"] = jump_index
        return
    mapped = map_mosaic_click(state["placements"], x, y)
    if mapped is None:
      return
    camera, point = mapped
    if event == cv2.EVENT_LBUTTONDOWN:
      state["observations"][camera] = point
      state["active_camera"] = camera
    elif event == cv2.EVENT_RBUTTONDOWN:
      state["observations"].pop(camera, None)
      state["active_camera"] = camera

  try:
    # Keep a 1:1 relationship between mosaic pixels and mouse coordinates.
    # Arbitrary window resizing would make small-target clicks inaccurate.
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
  except cv2.error as error:
    raise RuntimeError(
      "manual observation requires an OpenCV GUI environment"
    ) from error

  try:
    while 0 <= index < len(work_items):
      item = work_items[index]
      observation_name = item["frame"]
      source_frame = item["source_frame"]
      source_index = all_frame_names.index(source_frame)
      images = [
        cv2.imread(
          str(Path(camera_images.image_path) / camera_files[source_index]),
          cv2.IMREAD_COLOR
        )
        for camera_files in camera_images.filenames
      ]
      state["observations"] = {
        camera: list(point)
        for camera, point in annotations.get(observation_name, {}).items()
      }
      state["active_camera"] = (
        next(iter(state["observations"]))
        if len(state["observations"]) == 1 else None
      )

      while True:
        canvas, placements = compose_mosaic(
          camera_images.cameras,
          images,
          state["observations"],
          columns,
          tile_width
        )
        state["placements"] = placements
        canvas, sidebar_hitboxes = compose_point_sidebar(
          canvas,
          observation_names,
          annotations,
          index
        )
        state["sidebar_hitboxes"] = sidebar_hitboxes
        instruction = (
          "{}/{} image={} point={} | select>={} | "
          "left:set/refine right:remove "
          "arrows:1px | Enter:save N:skip B:back R:reset X:delete "
          "E/Q/Esc:quit"
        ).format(
          index + 1,
          "auto" if automatic_points else len(work_items),
          source_frame,
          item.get("point_name", observation_name),
          minimum_cameras
        )
        cv2.putText(
          canvas, instruction,
          (10, canvas.shape[0] - 18),
          cv2.FONT_HERSHEY_SIMPLEX, 0.55,
          (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(30)
        if state["jump_index"] is not None:
          index = state["jump_index"]
          state["jump_index"] = None
          break
        arrow_delta = None
        if key in (81, 2424832, 63234, 65361, ord("a"), ord("A")):
          arrow_delta = (-1.0, 0.0)
        elif key in (83, 2555904, 63235, 65363, ord("d"), ord("D")):
          arrow_delta = (1.0, 0.0)
        elif key in (82, 2490368, 63232, 65362, ord("w"), ord("W")):
          arrow_delta = (0.0, -1.0)
        elif key in (84, 2621440, 63233, 65364, ord("s"), ord("S")):
          arrow_delta = (0.0, 1.0)
        if arrow_delta is not None:
          nudge_observation(
            state["observations"],
            state["active_camera"],
            state["placements"],
            *arrow_delta
          )
          continue
        if key in (10, 13, 32):
          if len(state["observations"]) < minimum_cameras:
            continue
          annotations[observation_name] = {
            camera: list(point)
            for camera, point in state["observations"].items()
          }
          write_observations(
            output, image_path, camera_images.cameras,
            observation_names, annotations, minimum_cameras,
            source_frames
          )
          if automatic_points and index == len(work_items) - 1:
            append_automatic_point(work_items)
            next_item = work_items[-1]
            observation_names.append(next_item["frame"])
            source_frames[next_item["frame"]] = next_item["source_frame"]
          index += 1
          break
        if key in (ord("n"), ord("N")):
          if automatic_points and index == len(work_items) - 1:
            append_automatic_point(work_items)
            next_item = work_items[-1]
            observation_names.append(next_item["frame"])
            source_frames[next_item["frame"]] = next_item["source_frame"]
          index += 1
          break
        if key in (ord("b"), ord("B")):
          index = max(0, index - 1)
          break
        if key in (ord("r"), ord("R")):
          state["observations"].clear()
          state["active_camera"] = None
        if key in (ord("x"), ord("X")):
          annotations.pop(observation_name, None)
          write_observations(
            output, image_path, camera_images.cameras,
            observation_names, annotations, minimum_cameras,
            source_frames
          )
          index += 1
          break
        if key in (
            ord("e"), ord("E"), ord("q"), ord("Q"), 27):
          return write_observations(
            output, image_path, camera_images.cameras,
            observation_names, annotations, minimum_cameras,
            source_frames
          )
  finally:
    cv2.destroyWindow(window_name)

  return write_observations(
    output, image_path, camera_images.cameras,
    observation_names, annotations, minimum_cameras,
    source_frames
  )


@dataclass
class Observe:
  """Manually click synchronized target pixels in multiple cameras.

  With --frame and no --points, names are generated as P01, P02, ... until
  E, Q or Esc is used.
  """

  image_path: str
  cameras: List[str] = list_field()
  output: str = "object_observations.yaml"
  camera_pattern: Optional[str] = None
  columns: int = 2
  tile_width: int = 640
  start_frame: Optional[str] = None
  frame: Optional[str] = None
  points: List[str] = list_field()

  def execute(self):
    result, destination = annotate_observations(
      self.image_path,
      self.cameras,
      self.output,
      self.camera_pattern,
      self.columns,
      self.tile_width,
      self.start_frame,
      self.frame,
      self.points
    )
    print(
      "Saved {} annotated frames to {}".format(
        len(result["frames"]), destination
      )
    )


if __name__ == "__main__":
  run_with(Observe)
