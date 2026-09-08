"""Interactive court editor for world-coordinate YAML files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from simple_parsing import choice
from simple_parsing.helpers import list_field

from multical.config.arguments import run_with


COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE = 5.485

SNAP_ANCHORS_CONFIG = (
  Path(__file__).resolve().parent.parent / "config" / "court_snap_anchors.yaml"
)


def load_court_snap_anchors(court):
  """Load additional fixed XY anchors; court intersections remain automatic."""
  path = SNAP_ANCHORS_CONFIG
  try:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
      raise ValueError("expected a mapping of court names to XY lists")
    points = data.get(court, [])
    if not isinstance(points, list):
      raise ValueError("{} must be a list of [X, Y] coordinates".format(court))
    anchors = []
    seen = set()
    for index, point in enumerate(points):
      if not isinstance(point, list) or len(point) != 2:
        raise ValueError("{}[{}] must be [X, Y]".format(court, index))
      if any(isinstance(value, bool) or not isinstance(value, (int, float))
             or not np.isfinite(value) for value in point):
        raise ValueError("{}[{}] must contain finite numbers".format(court, index))
      x, y = map(float, point)
      if (x, y) not in seen:
        anchors.append({"x": x, "y": y})
        seen.add((x, y))
    return anchors
  except (OSError, yaml.YAMLError, ValueError) as error:
    raise ValueError("Cannot load fixed snap anchors from {}: {}".format(
      path, error)) from error


def court_geometry(court="tennis"):
  """Court guide in meters, with the origin at the near baseline center.

  Badminton dimensions: BWF Laws of Badminton, Diagram A.
  Tennis retains the existing editor's guide coordinates.
  """
  if court == "tennis":
    geometry = dict(length=COURT_LENGTH, doubles=DOUBLES_WIDTH,
                    singles=SINGLES_WIDTH, service=SERVICE_LINE,
                    net_center=.914, net_post=1.07, net_half=6.4,
                    net_bottom=.04, label="网球场")
  elif court == "badminton":
    geometry = dict(length=13.4, doubles=6.1, singles=5.18, service=4.72,
                    net_center=1.524, net_post=1.55, net_half=3.05,
                    net_bottom=.79, label="羽毛球场")
  else:
    raise ValueError("court must be tennis or badminton")
  length, width = geometry["length"], geometry["doubles"] / 2
  singles, service = geometry["singles"] / 2, geometry["service"]
  lines = [[0, y, length, y] for y in (-width, width, -singles, singles)]
  for x in (0, service, length / 2, length - service, length):
    half = width if court == "badminton" or x in (0, length) else singles
    # The badminton net is above the floor; there is no painted midcourt line.
    if court != "badminton" or x != length / 2:
      lines.append([x, -half, x, half])
  if court == "badminton":
    lines.extend([[.76, -width, .76, width],
                  [length-.76, -width, length-.76, width],
                  [0, 0, service, 0], [length-service, 0, length, 0]])
  else:
    lines.append([service, 0, length-service, 0])
  geometry["lines"] = lines
  geometry["snap_anchors"] = load_court_snap_anchors(court)
  return geometry


class _FlowList(list):
  """A list rendered on one YAML line (used for coordinates/cameras)."""


class _ConfigDumper(yaml.SafeDumper):
  pass


def _represent_flow_list(dumper, values):
  return dumper.represent_sequence(
    "tag:yaml.org,2002:seq", values, flow_style=True
  )


_ConfigDumper.add_representer(_FlowList, _represent_flow_list)


def _finite(value, description):
  number = float(value)
  if not np.isfinite(number):
    raise ValueError("{} must be finite".format(description))
  return number


def parse_marker_specs(marker_ids, marker_heights, marker_occurrences=""):
  """Parse aligned comma-separated marker IDs, heights and occurrences."""
  ids = [value.strip() for value in str(marker_ids).split(",") if value.strip()]
  heights = [
    value.strip() for value in str(marker_heights).split(",") if value.strip()
  ]
  occurrences = [
    value.strip() for value in str(marker_occurrences).split(",")
    if value.strip()
  ]
  if not ids:
    raise ValueError("at least one marker ID is required")
  if len(ids) != len(heights):
    raise ValueError("marker IDs and fixed heights must have equal counts")
  if occurrences and len(occurrences) != len(ids):
    raise ValueError("occurrences must be empty or match the marker count")

  specs = []
  for index, (marker_id, height) in enumerate(zip(ids, heights)):
    spec = {
      "marker_id": int(marker_id),
      "height": _finite(height, "marker height")
    }
    if occurrences:
      occurrence = occurrences[index].lower()
      if occurrence not in ("upper", "lower", "-"):
        raise ValueError("occurrence must be upper, lower, or -")
      if occurrence != "-":
        spec["occurrence"] = occurrence
    specs.append(spec)
  return specs


def load_marker_board(filename):
  """Load reusable marker IDs and center heights from a board YAML file."""
  path = Path(filename).expanduser().resolve()
  try:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  except (OSError, yaml.YAMLError) as error:
    raise ValueError("cannot load marker board {}: {}".format(path, error))
  if not isinstance(data, dict):
    raise ValueError("marker board must be a YAML mapping")
  markers = data.get("markers")
  if not isinstance(markers, list) or not markers:
    raise ValueError("marker board must define a non-empty markers list")

  specs = []
  for index, marker in enumerate(markers):
    if not isinstance(marker, dict):
      raise ValueError("marker board marker {} must be a mapping".format(index))
    if "marker_id" not in marker or "height" not in marker:
      raise ValueError(
        "marker board marker {} requires marker_id and height".format(index)
      )
    spec = {
      "marker_id": int(marker["marker_id"]),
      "height": _finite(marker["height"], "marker height")
    }
    occurrence = str(marker.get("occurrence", "")).strip().lower()
    if occurrence:
      if occurrence not in ("upper", "lower"):
        raise ValueError("marker occurrence must be upper or lower")
      spec["occurrence"] = occurrence
    specs.append(spec)

  marker_family = str(data.get("marker_family", "6X6_250")).strip()
  if not marker_family:
    raise ValueError("marker_family must not be empty")
  return marker_family, specs


def marker_configuration(options):
  """Resolve marker settings, preferring a reusable board YAML."""
  board = options.world_board or options.marker_board
  if board:
    return load_marker_board(board)
  return options.marker_family, parse_marker_specs(
    options.marker_ids, options.marker_heights, options.marker_occurrences
  )


def marker_specs_as_strings(specs):
  """Convert marker specs to the editable comma-separated UI values."""
  return (
    ",".join(str(spec["marker_id"]) for spec in specs),
    ",".join("{:.3f}".format(spec["height"]) for spec in specs),
    ",".join(spec.get("occurrence", "-") for spec in specs)
    if any(spec.get("occurrence") for spec in specs) else ""
  )


def measured_world_points_document(points, world_units="meters"):
  """Build the measured_world_points.yaml document."""
  result = {}
  for name, point in points:
    if not str(name).strip():
      raise ValueError("point names must not be empty")
    if str(name) in result:
      raise ValueError("duplicate point name {}".format(name))
    if len(point) != 3:
      raise ValueError("point {} must have X, Y and Z".format(name))
    result[str(name)] = _FlowList([
      round(_finite(value, "point coordinate"), 3) for value in point
    ])
  return {
    "coordinate_frame": "world",
    "world_units": str(world_units),
    "points": result
  }


def world_markers_document(
    locations, cameras, marker_specs, marker_family="6X6_250",
    image_path="world_images", world_units="meters"):
  """Build a marker-center world correspondence document."""
  if not cameras:
    raise ValueError("at least one camera is required")
  if not marker_specs:
    raise ValueError("at least one marker definition is required")
  captures = []
  names = set()
  for name, x, y in locations:
    capture_name = str(name)
    if not capture_name:
      raise ValueError("capture names must not be empty")
    if capture_name in names:
      raise ValueError("duplicate capture name {}".format(capture_name))
    names.add(capture_name)
    markers = []
    for spec in marker_specs:
      marker = {"marker_id": int(spec["marker_id"])}
      if spec.get("occurrence"):
        marker["occurrence"] = str(spec["occurrence"])
      marker["world_point"] = _FlowList([
        round(_finite(x, "X coordinate"), 3),
        round(_finite(y, "Y coordinate"), 3),
        round(_finite(spec["height"], "marker height"), 3)
      ])
      markers.append(marker)
    captures.append({"name": capture_name, "markers": markers})
  return {
    "world_units": str(world_units),
    "cameras": _FlowList([str(camera) for camera in cameras]),
    "marker_family": str(marker_family),
    "image_path": str(image_path),
    "marker_quality": {
      "mode": "reject",
      "min_edge_px": 15,
      "warn_edge_px": 25,
      "min_side_ratio": 0.20,
      "min_area_ratio": 0.15,
      "max_view_angle_deg": 60,
      "warn_view_angle_deg": 45
    },
    "captures": captures
  }


def write_world_config(filename, document):
  destination = Path(filename).expanduser().resolve()
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(
    yaml.dump(
      document, Dumper=_ConfigDumper, sort_keys=False,
      allow_unicode=True, default_flow_style=False
    ),
    encoding="utf-8"
  )
  return destination


def load_observe_points(filename):
  """Read point names, source frames and clicked pixels from observe YAML."""
  path = Path(filename).expanduser().resolve()
  data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  if not isinstance(data, dict):
    raise ValueError("observe file must be a YAML mapping")
  source = data.get("source") or {}
  image_path = source.get("image_path", path.parent)
  image_root = Path(image_path).expanduser()
  if not image_root.is_absolute():
    image_root = (path.parent / image_root).resolve()
  else:
    image_root = image_root.resolve()
  # Observe outputs store an absolute source path. When a complete dataset is
  # copied/renamed for an experiment, prefer the colocated image directory so
  # the copy is self-contained without rewriting the read-only observe YAML.
  local_image_root = (path.parent / image_root.name).resolve()
  if local_image_root.is_dir() and local_image_root != image_root:
    image_root = local_image_root
  points = {}

  frames = data.get("frames")
  if isinstance(frames, list):
    for index, frame in enumerate(frames):
      if not isinstance(frame, dict) or frame.get("frame") is None:
        raise ValueError("observe frame {} is invalid".format(index))
      name = str(frame.get("point_name", frame["frame"]))
      observations = frame.get("observations") or {}
      if not isinstance(observations, dict):
        raise ValueError("observe point {} observations must be a map".format(name))
      points[name] = {
        "source_frame": str(frame.get("source_frame", frame["frame"])),
        "observations": {
          str(camera): [float(pixel[0]), float(pixel[1])]
          for camera, pixel in observations.items()
        }
      }

  observations = data.get("observations")
  if isinstance(observations, list):
    for index, observation in enumerate(observations):
      if not isinstance(observation, dict):
        raise ValueError("observe observation {} is invalid".format(index))
      name = observation.get("annotation_id", observation.get("capture"))
      camera = observation.get("camera")
      pixel = observation.get("image_point")
      if name is None or camera is None or pixel is None:
        raise ValueError(
          "observe observation {} needs capture, camera and image_point".format(
            index
          )
        )
      name = str(name)
      point = points.setdefault(name, {
        "source_frame": str(observation.get("source_frame", name)),
        "observations": {}
      })
      point["observations"][str(camera)] = [float(pixel[0]), float(pixel[1])]

  if not points:
    raise ValueError("observe file contains no annotated points")
  source_frames = [point["source_frame"] for point in points.values()]
  unique_source_frames = set(source_frames)
  if len(points) > 1 and len(unique_source_frames) == 1:
    selection_mode = "single_image_multiple_points"
  elif len(unique_source_frames) == len(points):
    selection_mode = "multiple_images_single_point"
  else:
    selection_mode = "mixed"
  cameras = [str(camera) for camera in data.get("cameras", [])]
  if not cameras:
    cameras = list(dict.fromkeys(
      camera
      for point in points.values()
      for camera in point["observations"]
    ))
  return {
    "path": path,
    "image_root": image_root,
    "cameras": cameras,
    "points": points,
    "selection_mode": selection_mode
  }


def find_observe_image(image_root, camera, source_frame):
  """Resolve an observe source image without modifying the source tree."""
  root = Path(image_root)
  if root.is_file():
    return root
  frame = str(source_frame)
  candidates = [root / str(camera) / frame, root / frame]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  matches = list(root.rglob(frame)) if root.is_dir() else []
  camera_matches = [
    match for match in matches if str(camera) in match.parts
  ]
  return (camera_matches or matches or [None])[0]


def _draw_court_2d(axis, court="tennis"):
  geometry = court_geometry(court)
  length, width = geometry["length"], geometry["doubles"]
  half_doubles = width / 2.0
  white = "#f8fafc"
  axis.set_facecolor("#256a4a")
  for x0, y0, x1, y1 in geometry["lines"]:
    axis.plot([x0, x1], [y0, y1], color=white, linewidth=1.5)
  axis.set_xlim(-0.8, length + 0.8)
  axis.set_ylim(-half_doubles - 0.8, half_doubles + 0.8)
  axis.set_aspect("equal", adjustable="box")
  axis.set_xlabel("X / m")
  axis.set_ylabel("Y / m")
  axis.set_title("Top view: click to add XY")
  axis.grid(alpha=0.18)


def _draw_court_3d(axis, maximum_height, court="tennis"):
  geometry = court_geometry(court)
  length, width = geometry["length"], geometry["doubles"]
  half_doubles = width / 2.0
  white = "#f8fafc"
  axis.set_facecolor("#eef5f1")
  surface_x, surface_y = np.meshgrid(
    [0, length], [-half_doubles, half_doubles]
  )
  axis.plot_surface(
    surface_x, surface_y, np.zeros_like(surface_x),
    color="#256a4a", alpha=.88, shade=False
  )
  for x0, y0, x1, y1 in geometry["lines"]:
    axis.plot([x0, x1], [y0, y1], [0, 0], color=white, linewidth=2)
  axis.plot(
    [length / 2.0] * 3,
    [-geometry["net_half"], 0, geometry["net_half"]],
    [geometry["net_post"], geometry["net_center"], geometry["net_post"]],
    color="#334155", linewidth=2
  )
  axis.quiver(0, 0, 0, 3, 0, 0, color="#dc2626", arrow_length_ratio=.12)
  axis.quiver(0, 0, 0, 0, 3, 0, color="#16a34a", arrow_length_ratio=.12)
  axis.quiver(0, 0, 0, 0, 0, 1.5, color="#2563eb", arrow_length_ratio=.12)
  axis.text(3.2, 0, 0, "X", color="#dc2626")
  axis.text(0, 3.2, 0, "Y", color="#16a34a")
  axis.text(0, 0, 1.65, "Z", color="#2563eb")
  axis.set_xlim(0, length)
  axis.set_ylim(-half_doubles, half_doubles)
  axis.set_zlim(0, max(2.2, maximum_height + 0.4))
  axis.set_xlabel("X / m")
  axis.set_ylabel("Y / m")
  axis.set_zlabel("Z / m")
  axis.set_title("3D world-coordinate preview")
  axis.view_init(elev=24, azim=-62)
  try:
    axis.set_box_aspect((length, width, 5.0))
  except AttributeError:
    pass


class WorldPointEditor:
  """Matplotlib-based editor; imports GUI modules only when launched."""

  def __init__(self, options):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    self.plt = plt
    self.court = getattr(options, "court", "tennis")
    self.mode = options.mode
    self.output = options.output or (
      "measured_world_points.yaml" if self.mode == "measured"
      else "world_markers.yaml"
    )
    self.world_units = options.world_units
    self.cameras = list(options.cameras)
    self.marker_family, configured_marker_specs = marker_configuration(options)
    marker_ids, marker_heights, marker_occurrences = marker_specs_as_strings(
      configured_marker_specs
    )
    self.image_path = options.image_path
    self.point_prefix = options.point_prefix
    self.capture_start = options.capture_start
    self.capture_digits = options.capture_digits
    self.snap = options.snap
    self.points = []
    self.locations = []
    self.selected = None
    self.pending_xy = False
    self.observe_data = (
      load_observe_points(options.observe)
      if self.mode == "measured" and options.observe else None
    )
    self.observation_names = (
      list(self.observe_data["points"]) if self.observe_data else []
    )
    self.active_observation = None
    self.list_offset = 0
    self.visible_point_names = []
    self.observation_figure = None
    self.observation_axis = None
    if self.mode == "measured" and Path(self.output).expanduser().is_file():
      existing = yaml.safe_load(
        Path(self.output).expanduser().read_text(encoding="utf-8")
      ) or {}
      for name, point in (existing.get("points") or {}).items():
        if len(point) == 3:
          self.points.append((str(name), tuple(float(value) for value in point)))

    self.figure = plt.figure(figsize=(15, 8))
    self.figure.canvas.manager.set_window_title("World Coordinate Editor")
    self.ax3d = self.figure.add_axes([.04, .12, .48, .80], projection="3d")
    self.axtop = self.figure.add_axes([.55, .51, .25, .41])
    self.axside = self.figure.add_axes([.55, .12, .25, .27])
    self.status = self.figure.text(.04, .035, "", fontsize=9)
    self.summary_axis = self.figure.add_axes([.84, .33, .14, .17])
    self.summary_axis.set_axis_off()
    self.summary = self.summary_axis.text(
      0, 1, "", fontsize=8, va="top", clip_on=True,
      transform=self.summary_axis.transAxes
    )
    self.point_row_artists = []

    box_x, box_w, box_h = .87, .11, .045
    self.name_box = TextBox(
      self.figure.add_axes([box_x, .86, box_w, box_h]),
      "Name ", initial=self._next_name()
    )
    self.x_box = TextBox(
      self.figure.add_axes([box_x, .80, box_w, box_h]), "X ", initial="0.000"
    )
    self.y_box = TextBox(
      self.figure.add_axes([box_x, .74, box_w, box_h]), "Y ", initial="0.000"
    )
    self.z_box = TextBox(
      self.figure.add_axes([box_x, .68, box_w, box_h]),
      "Z(s) " if self.mode == "markers" else "Z ",
      initial=str(marker_heights if self.mode == "markers" else 0.0)
    )
    self.ids_box = None
    self.occurrences_box = None
    if self.mode == "markers":
      self.ids_box = TextBox(
        self.figure.add_axes([box_x, .62, box_w, box_h]),
        "IDs ", initial=marker_ids
      )
      self.occurrences_box = TextBox(
        self.figure.add_axes([box_x, .56, box_w, box_h]),
        "Occ. ", initial=marker_occurrences
      )

    add_button = Button(
      self.figure.add_axes([.84, .26, .065, .05]), "Add/Update"
    )
    undo_button = Button(
      self.figure.add_axes([.915, .26, .065, .05]), "Undo"
    )
    save_button = Button(
      self.figure.add_axes([.84, .19, .065, .05]), "Save"
    )
    clear_button = Button(
      self.figure.add_axes([.915, .19, .065, .05]), "Clear"
    )
    add_button.on_clicked(self._add_or_update)
    undo_button.on_clicked(self._undo)
    save_button.on_clicked(self._save)
    clear_button.on_clicked(self._clear)
    self._widgets = [add_button, undo_button, save_button, clear_button]
    self.figure.canvas.mpl_connect("button_press_event", self._on_click)
    self.figure.canvas.mpl_connect("key_press_event", self._on_key)
    if self.observation_names:
      self.active_observation = self._next_name()
      self.name_box.set_val(self.active_observation)
      self._refresh("Click a point in the list to show its observe result")
    else:
      self._refresh("Click the top view to add a point")

  def _next_name(self):
    if self.observation_names:
      assigned = {name for name, _point in self.points}
      return next(
        (name for name in self.observation_names if name not in assigned),
        self.observation_names[0]
      )
    count = len(self.points) if self.mode == "measured" else len(self.locations)
    if self.mode == "measured":
      return "{}{:02d}".format(self.point_prefix, count + 1)
    return str(self.capture_start + count).zfill(self.capture_digits)

  def _rounded(self, value):
    value = _finite(value, "coordinate")
    if self.snap > 0:
      value = round(value / self.snap) * self.snap
    return round(value, 3)

  def _specs(self):
    return parse_marker_specs(
      self.ids_box.text, self.z_box.text, self.occurrences_box.text
    )

  def _on_click(self, event):
    if event.inaxes == self.summary_axis and event.button in ("up", "down"):
      direction = -1 if event.button == "up" else 1
      maximum = max(0, len(self.observation_names) - 6)
      self.list_offset = min(maximum, max(0, self.list_offset + direction))
      self._update_point_list()
      self.figure.canvas.draw_idle()
      return
    if event.button not in (1, 3) or event.xdata is None or event.ydata is None:
      return
    toolbar = getattr(self.figure.canvas, "toolbar", None)
    if toolbar is not None and getattr(toolbar, "mode", ""):
      return
    if event.inaxes == self.summary_axis and event.button == 1:
      row = min(
        range(len(self.visible_point_names)),
        key=lambda index: abs(event.ydata - (.72 - index * .115)),
        default=None
      )
      if row is not None and abs(event.ydata - (.72 - row * .115)) <= .055:
        self._select_observation(self.visible_point_names[row])
      return
    if event.inaxes == self.axtop:
      if event.button == 3:
        self._select_nearest(event.xdata, event.ydata)
        return
      self.x_box.set_val("{:.3f}".format(self._rounded(event.xdata)))
      self.y_box.set_val("{:.3f}".format(self._rounded(event.ydata)))
      self.name_box.set_val(self.active_observation or self._next_name())
      if self.active_observation:
        self.selected = next(
          (index for index, point in enumerate(self.points)
           if point[0] == self.active_observation),
          None
        )
      else:
        self.selected = None
      if self.mode == "measured":
        self.pending_xy = True
        self._refresh(
          "{} XY selected. Click the side view to set Z".format(
            self.name_box.text.strip()
          )
        )
      else:
        self._add_or_update(None)
    elif event.inaxes == self.axside and self.mode == "measured":
      self.z_box.set_val("{:.3f}".format(max(0.0, self._rounded(event.ydata))))
      if self.pending_xy or self.selected is not None:
        point_name = self.name_box.text.strip()
        self._refresh(
          "{} Z selected. Click Add/Update to save".format(point_name)
        )
      else:
        self._refresh("Select XY in the top view first")

  def _select_nearest(self, x, y):
    records = self.points if self.mode == "measured" else self.locations
    if not records:
      return
    xy_values = np.asarray([
      (item[1][0], item[1][1]) if self.mode == "measured"
      else (item[1], item[2])
      for item in records
    ])
    distances = np.linalg.norm(xy_values - np.asarray([x, y]), axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 1.0:
      self._refresh("No point within 1 m of the right-click")
      return
    if records[index][0] in self.observation_names:
      self._select_observation(records[index][0])
      return
    self.selected = index
    self.pending_xy = False
    item = records[index]
    self.name_box.set_val(item[0])
    if self.mode == "measured":
      self.x_box.set_val("{:.3f}".format(item[1][0]))
      self.y_box.set_val("{:.3f}".format(item[1][1]))
      self.z_box.set_val("{:.3f}".format(item[1][2]))
    else:
      self.x_box.set_val("{:.3f}".format(item[1]))
      self.y_box.set_val("{:.3f}".format(item[2]))
    self._refresh("Selected {}. Edit fields, then Add/Update".format(item[0]))

  def _select_observation(self, name):
    if name not in self.observation_names:
      return
    self.active_observation = name
    self.pending_xy = False
    self.name_box.set_val(name)
    matching = [
      (index, point) for index, point in enumerate(self.points)
      if point[0] == name
    ]
    if matching:
      self.selected, item = matching[0]
      self.x_box.set_val("{:.3f}".format(item[1][0]))
      self.y_box.set_val("{:.3f}".format(item[1][1]))
      self.z_box.set_val("{:.3f}".format(item[1][2]))
    else:
      self.selected = None
      self.x_box.set_val("0.000")
      self.y_box.set_val("0.000")
      self.z_box.set_val("0.000")
    index = self.observation_names.index(name)
    if index < self.list_offset:
      self.list_offset = index
    elif index >= self.list_offset + 6:
      self.list_offset = index - 5
    self._refresh(
      "Selected {}. Set XYZ, then click Add/Update".format(name)
    )
    self._show_observe_result(name)

  def _show_observe_result(self, name):
    import cv2

    from multical.app.observe import compose_mosaic

    point = self.observe_data["points"][name]
    cameras = []
    images = []
    annotations = {}
    for camera in self.observe_data["cameras"]:
      if camera not in point["observations"]:
        continue
      image_file = find_observe_image(
        self.observe_data["image_root"], camera, point["source_frame"]
      )
      image = cv2.imread(str(image_file)) if image_file is not None else None
      if image is None:
        continue
      cameras.append(camera)
      images.append(image)
      annotations[camera] = point["observations"][camera]
    if not cameras:
      self.status.set_text(
        "Selected {}, but its observe source images were not found under {}".format(
          name, self.observe_data["image_root"]
        )
      )
      self.figure.canvas.draw_idle()
      return
    mosaic, _placements = compose_mosaic(
      cameras, images, annotations,
      columns=min(2, len(cameras)), tile_width=520
    )
    if self.observation_figure is None:
      self.observation_figure = self.plt.figure(figsize=(10, 7))
      self.observation_axis = self.observation_figure.add_axes([0, 0, 1, 1])
      self.observation_axis.set_axis_off()
    self.observation_figure.canvas.manager.set_window_title(
      "Observe result - {}".format(name)
    )
    self.observation_axis.clear()
    self.observation_axis.set_axis_off()
    self.observation_axis.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    self.observation_axis.set_title(
      "{} - read-only observe clicks".format(name), pad=8
    )
    self.observation_figure.canvas.draw_idle()
    if str(self.plt.get_backend()).lower() != "agg":
      self.observation_figure.show()

  def _next_observation(self, name):
    if name not in self.observation_names:
      return None
    index = self.observation_names.index(name) + 1
    return (
      self.observation_names[index]
      if index < len(self.observation_names) else None
    )

  def _add_or_update(self, _event):
    try:
      name = self.name_box.text.strip()
      x = self._rounded(self.x_box.text)
      y = self._rounded(self.y_box.text)
      if self.mode == "measured":
        z = self._rounded(self.z_box.text)
        record = (name, (x, y, z))
        records = self.points
      else:
        self._specs()
        record = (name, x, y)
        records = self.locations
      if not name:
        raise ValueError("name must not be empty")
      if self.observation_names and name not in self.observation_names:
        raise ValueError("point {} is not present in the observe file".format(name))
      duplicate = next(
        (index for index, item in enumerate(records)
         if item[0] == name and index != self.selected), None
      )
      if duplicate is not None:
        raise ValueError("name {} already exists".format(name))
      if self.selected is None:
        records.append(record)
        self.selected = len(records) - 1
      else:
        records[self.selected] = record
      self._write()
      saved_name = name
      self.selected = None
      self.pending_xy = False
      next_observation = self._next_observation(saved_name)
      if next_observation is not None:
        self._select_observation(next_observation)
        self.status.set_text(
          "Saved {}. Selected {}".format(saved_name, next_observation)
        )
        self.figure.canvas.draw_idle()
      else:
        self.active_observation = None
        self.name_box.set_val(self._next_name())
        self._refresh("Saved {}".format(self.output))
    except (TypeError, ValueError) as error:
      self._refresh("Error: {}".format(error))

  def _document(self):
    if self.mode == "measured":
      points = self.points
      if self.observation_names:
        point_map = dict(self.points)
        points = [
          (name, point_map[name]) for name in self.observation_names
          if name in point_map
        ]
      return measured_world_points_document(points, self.world_units)
    return world_markers_document(
      self.locations, self.cameras, self._specs(), self.marker_family,
      self.image_path, self.world_units
    )

  def _write(self):
    write_world_config(self.output, self._document())

  def _save(self, _event):
    try:
      self._write()
      self._refresh("Saved {}".format(Path(self.output).resolve()))
    except (TypeError, ValueError) as error:
      self._refresh("Error: {}".format(error))

  def _undo(self, _event):
    records = self.points if self.mode == "measured" else self.locations
    if records:
      records.pop()
      self.selected = None
      self.pending_xy = False
      self._write()
      self.name_box.set_val(self._next_name())
      self._refresh("Removed last point and saved")

  def _clear(self, _event):
    records = self.points if self.mode == "measured" else self.locations
    records[:] = []
    self.selected = None
    self.pending_xy = False
    self._write()
    self.name_box.set_val(self._next_name())
    self._refresh("Cleared all points and saved")

  def _on_key(self, event):
    if event.key in ("ctrl+z", "cmd+z"):
      self._undo(None)
    elif event.key in ("ctrl+s", "cmd+s"):
      self._save(None)

  def _refresh(self, message):
    maximum_height = 0.0
    if self.mode == "measured" and self.points:
      maximum_height = max(point[1][2] for point in self.points)
    elif self.mode == "markers":
      try:
        maximum_height = max(spec["height"] for spec in self._specs())
      except ValueError:
        pass
    self.ax3d.clear()
    self.axtop.clear()
    self.axside.clear()
    _draw_court_3d(self.ax3d, maximum_height, self.court)
    _draw_court_2d(self.axtop, self.court)
    self.axside.set_facecolor("#f1f5f9")
    self.axside.set_xlim(-0.8, court_geometry(self.court)["length"] + 0.8)
    self.axside.set_ylim(0, max(2.2, maximum_height + .4))
    self.axside.set_xlabel("X / m")
    self.axside.set_ylabel("Z / m")
    self.axside.set_title(
      "Side view: click to set current Z"
      if self.mode == "measured" else "Side elevation preview"
    )
    self.axside.grid(alpha=.25)
    if self.mode == "measured":
      coordinates = [point for _, point in self.points]
      names = [name for name, _ in self.points]
    else:
      coordinates = []
      names = []
      try:
        for name, x, y in self.locations:
          for spec in self._specs():
            coordinates.append((x, y, spec["height"]))
            names.append("{}:{}".format(name, spec["marker_id"]))
      except ValueError:
        pass
    if coordinates:
      values = np.asarray(coordinates)
      self.ax3d.scatter(
        values[:, 0], values[:, 1], values[:, 2], c="#f97316", s=35,
        depthshade=False
      )
      self.axtop.scatter(values[:, 0], values[:, 1], c="#f97316", s=28)
      self.axside.scatter(values[:, 0], values[:, 2], c="#f97316", s=28)
      for name, (x, y, z) in zip(names, coordinates):
        self.ax3d.text(x, y, z + .06, name, fontsize=7)
    if self.mode == "measured" and self.pending_xy:
      try:
        pending_x = self._rounded(self.x_box.text)
        pending_y = self._rounded(self.y_box.text)
        pending_z = max(0.0, self._rounded(self.z_box.text))
        self.axtop.scatter(
          [pending_x], [pending_y], marker="x", c="#22d3ee", s=70,
          linewidths=2
        )
        self.axside.scatter(
          [pending_x], [pending_z], marker="x", c="#22d3ee", s=70,
          linewidths=2
        )
      except ValueError:
        pass
    self._update_point_list()
    self.status.set_text(message)
    self.figure.canvas.draw_idle()

  def _update_point_list(self):
    for artist in self.point_row_artists:
      artist.remove()
    self.point_row_artists = []
    if self.observation_names:
      assigned = {name for name, _point in self.points}
      self.visible_point_names = self.observation_names[
        self.list_offset:self.list_offset + 6
      ]
      self.summary.set_text("Observe points ({}/{})\n(scroll list)".format(
        len(assigned.intersection(self.observation_names)),
        len(self.observation_names)
      ))
      for index, name in enumerate(self.visible_point_names):
        selected = ">" if name == self.active_observation else " "
        completed = "[x]" if name in assigned else "[ ]"
        artist = self.summary_axis.text(
          0, .72 - index * .115,
          "{} {} {}".format(selected, completed, name),
          fontsize=8, va="center", clip_on=True,
          transform=self.summary_axis.transAxes,
          color="#1d4ed8" if name == self.active_observation else "#111827"
        )
        self.point_row_artists.append(artist)
      return

    records = self.points if self.mode == "measured" else self.locations
    recent = records[-5:]
    self.visible_point_names = []
    summary = ["{} points".format(len(records)), ""]
    if len(records) > len(recent):
      summary.append("... {} earlier".format(len(records) - len(recent)))
    for item in recent:
      if self.mode == "measured":
        summary.append("{} [{:.3f}, {:.3f}, {:.3f}]".format(
          item[0], *item[1]
        ))
      else:
        summary.append("{} [{:.3f}, {:.3f}]".format(*item))
    self.summary.set_text("\n".join(summary))

  def show(self):
    self.plt.show()


@dataclass
class Worldpoints:
  """Click a 3D court model to generate world-coordinate YAML."""

  court: str = choice("tennis", "badminton", default="tennis")
  ui: str = choice("web", "desktop", default="web")
  mode: str = choice("measured", "markers", default="measured")
  output: Optional[str] = None
  observe: Optional[str] = None
  world_units: str = "meters"
  cameras: List[str] = list_field()
  marker_family: str = "6X6_250"
  world_board: Optional[str] = None
  # Deprecated compatibility alias; prefer world_board.
  marker_board: Optional[str] = None
  image_path: str = "world_images"
  marker_ids: str = "23,23"
  marker_heights: str = "1.700,0.500"
  marker_occurrences: str = "upper,lower"
  point_prefix: str = "P"
  capture_start: int = 0
  capture_digits: int = 6
  snap: float = 0.001
  host: str = "127.0.0.1"
  port: int = 8765
  no_browser: bool = False

  def execute(self):
    court_geometry(self.court)
    if self.mode == "markers" and self.observe is not None:
      raise ValueError("--observe is only available in measured mode")
    if self.mode == "markers" and self.ui == "desktop" and not self.cameras:
      raise ValueError(
        "marker mode requires --cameras, for example --cameras cam0 cam1"
      )
    if self.capture_digits < 1:
      raise ValueError("capture_digits must be at least 1")
    if self.snap < 0:
      raise ValueError("snap must not be negative")
    if not 0 <= self.port <= 65535:
      raise ValueError("port must be between 0 and 65535")
    if self.ui == "desktop":
      WorldPointEditor(self).show()
    else:
      from multical.app.worldpoints_web import serve_worldpoints
      serve_worldpoints(self)


if __name__ == "__main__":
  run_with(Worldpoints)
