"""Local web UI for the world-coordinate editor."""

from copy import deepcopy
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

import cv2
import yaml

from multical.app.observe import compose_mosaic
from multical.app.worldpoints import (
  court_geometry,
  find_observe_image,
  load_observe_points,
  marker_configuration,
  measured_world_points_document,
  parse_marker_specs,
  world_markers_document,
  write_world_config,
)


STATIC_ROOT = Path(__file__).resolve().parent.parent / "web" / "worldpoints"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def _natural_key(value):
  return [
    int(part) if part.isdigit() else part.lower()
    for part in re.split(r"([0-9]+)", str(value))
  ]


def load_marker_image_points(output, image_path, cameras=None):
  """Infer marker camera groups and captures from a world_images directory."""
  destination = Path(output).expanduser().resolve()
  configured = Path(image_path).expanduser()
  candidates = (
    [configured] if configured.is_absolute()
    else [destination.parent / configured, Path.cwd() / configured]
  )
  image_root = next(
    (candidate.resolve() for candidate in candidates if candidate.is_dir()),
    None
  )
  if image_root is None:
    raise ValueError(
      "world marker image directory was not found: {}".format(image_path)
    )
  selected_cameras = [str(camera) for camera in (cameras or [])]
  if not selected_cameras:
    match = re.search(
      r"world_markers_([0-9]+)\.ya?ml$", destination.name,
      flags=re.IGNORECASE
    )
    if match:
      selected_cameras = ["cam{}".format(number) for number in match.group(1)]
    else:
      selected_cameras = sorted(
        [directory.name for directory in image_root.iterdir()
         if directory.is_dir() and directory.name.startswith("cam")],
        key=_natural_key
      )
  missing = [
    camera for camera in selected_cameras
    if not (image_root / camera).is_dir()
  ]
  if missing:
    raise ValueError(
      "marker image camera directories are missing: {}".format(
        ", ".join(missing)
      )
    )
  captures = {}
  for camera in selected_cameras:
    for filename in sorted((image_root / camera).iterdir(), key=_natural_key):
      if not filename.is_file() or filename.suffix.lower() not in IMAGE_SUFFIXES:
        continue
      capture = captures.setdefault(filename.stem, {"files": {}})
      capture["files"][camera] = filename
  if not captures:
    raise ValueError("no marker capture images were found in {}".format(image_root))
  captures = {
    name: captures[name] for name in sorted(captures, key=_natural_key)
  }
  try:
    config_image_path = str(image_root.relative_to(destination.parent))
  except ValueError:
    config_image_path = str(image_path)
  return {
    "image_root": image_root,
    "cameras": selected_cameras,
    "captures": captures,
    "config_image_path": config_image_path
  }


class WorldpointsWebState:
  """Mutable editor state with all filesystem writes confined to output."""

  def __init__(self, options):
    self.options = options
    self.mode = options.mode
    self.court = court_geometry(getattr(options, "court", "tennis"))
    self.output = Path(options.output or (
      "measured_world_points.yaml" if self.mode == "measured"
      else "world_markers.yaml"
    )).expanduser().resolve()
    self.observe = None
    self.observe_error = None
    if self.mode == "measured" and options.observe:
      try:
        self.observe = load_observe_points(options.observe)
      except FileNotFoundError as error:
        if not self.output.is_file():
          raise
        self.observe_error = str(error)
    self.measured_yaml_only = bool(
      self.mode == "measured" and not self.observe
      and self.output.is_file()
    )
    self.marker_images = None
    self.marker_image_error = None
    if self.mode == "markers":
      try:
        self.marker_images = load_marker_image_points(
          self.output, options.image_path, options.cameras
        )
      except ValueError as error:
        if not self.output.is_file():
          raise
        self.marker_image_error = str(error)
    self.marker_yaml_only = bool(
      self.mode == "markers" and not self.marker_images
      and self.output.is_file()
    )
    self.cameras = (
      self.marker_images["cameras"]
      if self.marker_images else list(options.cameras)
    )
    self.marker_family, self.marker_specs = marker_configuration(options)
    self.config_image_path = (
      self.marker_images["config_image_path"]
      if self.marker_images else options.image_path
    )
    self.points = {}
    self.locations = []
    self.history = []
    self._load_output()
    self.snap_anchors = self._loaded_marker_anchors()

  def _loaded_marker_anchors(self):
    """Keep the marker file's original XY values as persistent snap targets."""
    if self.mode != "markers":
      return []
    anchors = []
    seen = set()
    for location in self.locations:
      key = (round(location["x"], 9), round(location["y"], 9))
      if key in seen:
        continue
      seen.add(key)
      anchors.append({"x": location["x"], "y": location["y"]})
    return anchors

  def _load_output(self):
    if not self.output.is_file():
      return
    data = yaml.safe_load(self.output.read_text(encoding="utf-8")) or {}
    if self.mode == "measured":
      for name, value in (data.get("points") or {}).items():
        if isinstance(value, list) and len(value) == 3:
          self.points[str(name)] = [float(item) for item in value]
      return
    if self.marker_yaml_only:
      self.cameras = [str(camera) for camera in (data.get("cameras") or [])]
      self.config_image_path = str(
        data.get("image_path") or self.config_image_path
      )
    self.marker_family = str(
      data.get("marker_family") or self.marker_family
    )
    for capture in data.get("captures") or []:
      markers = capture.get("markers") or []
      if not markers:
        continue
      point = markers[0].get("world_point") or []
      if len(point) == 3:
        self.locations.append({
          "name": str(capture.get("name", "")),
          "x": float(point[0]),
          "y": float(point[1])
        })
    captures = data.get("captures") or []
    if captures and captures[0].get("markers"):
      self.marker_specs = [
        {
          "marker_id": int(marker["marker_id"]),
          "height": float(marker["world_point"][2]),
          **({"occurrence": str(marker["occurrence"])}
             if marker.get("occurrence") else {})
        }
        for marker in captures[0]["markers"]
      ]

  def _snapshot(self):
    self.history.append((
      deepcopy(self.points), deepcopy(self.locations),
      deepcopy(self.marker_specs)
    ))
    self.history = self.history[-50:]

  def point_names(self):
    if self.observe:
      return list(self.observe["points"])
    if self.mode == "measured":
      names = list(self.points)
      number = 1
      while "{}{:02d}".format(self.options.point_prefix, number) in self.points:
        number += 1
      names.append("{}{:02d}".format(self.options.point_prefix, number))
      return names
    if self.marker_images:
      return list(self.marker_images["captures"])
    names = [location["name"] for location in self.locations]
    if self.marker_yaml_only and names:
      return names
    number = self.options.capture_start + len(names)
    candidate = str(number).zfill(self.options.capture_digits)
    while candidate in names:
      number += 1
      candidate = str(number).zfill(self.options.capture_digits)
    names.append(candidate)
    return names

  def payload(self):
    return {
      "mode": self.mode,
      "court": self.court,
      "output": str(self.output),
      "observe": str(self.observe["path"]) if self.observe else None,
      "observe_mode": (
        self.observe["selection_mode"] if self.observe else None
      ),
      "image_mode": (
        "observe" if self.observe else
        "measured_yaml" if self.measured_yaml_only else
        "marker_captures" if self.marker_images else
        "marker_yaml" if self.marker_yaml_only else None
      ),
      "has_images": bool(self.observe or self.marker_images),
      "image_warning": self.observe_error or self.marker_image_error,
      "point_names": self.point_names(),
      "points": self.points,
      "locations": self.locations,
      "snap_anchors": self.snap_anchors,
      "marker_specs": self.marker_specs,
      "cameras": list(self.cameras),
      "marker_family": self.marker_family,
      "image_path": self.config_image_path,
      "can_undo": bool(self.history)
    }

  @staticmethod
  def _number(value, name):
    try:
      result = float(value)
    except (TypeError, ValueError):
      raise ValueError("{} must be a number".format(name))
    if result != result or result in (float("inf"), float("-inf")):
      raise ValueError("{} must be finite".format(name))
    return round(result, 3)

  def save(self, data):
    name = str(data.get("name", "")).strip()
    if not name:
      raise ValueError("point name is required")
    self._snapshot()
    try:
      if self.mode == "measured":
        if self.observe and name not in self.observe["points"]:
          raise ValueError("{} is not present in observe".format(name))
        self.points[name] = [
          self._number(data.get("x"), "X"),
          self._number(data.get("y"), "Y"),
          self._number(data.get("z"), "Z")
        ]
      else:
        specs = parse_marker_specs(
          data.get("marker_ids", ""), data.get("marker_heights", ""),
          data.get("marker_occurrences", "")
        )
        self.marker_specs = specs
        location = {
          "name": name,
          "x": self._number(data.get("x"), "X"),
          "y": self._number(data.get("y"), "Y")
        }
        matching = next((
          index for index, item in enumerate(self.locations)
          if item["name"] == name
        ), None)
        if matching is None:
          self.locations.append(location)
        else:
          self.locations[matching] = location
      self.write()
    except Exception:
      self.points, self.locations, self.marker_specs = self.history.pop()
      raise
    return self.payload()

  def write(self):
    if self.mode == "measured":
      names = self.point_names() if self.observe else list(self.points)
      document = measured_world_points_document([
        (name, self.points[name]) for name in names if name in self.points
      ], self.options.world_units)
    else:
      document = world_markers_document(
        [(item["name"], item["x"], item["y"])
         for item in self.locations],
        self.cameras, self.marker_specs,
        self.marker_family,
        self.config_image_path,
        self.options.world_units
      )
    write_world_config(self.output, document)

  def undo(self):
    if self.history:
      self.points, self.locations, self.marker_specs = self.history.pop()
      self.write()
    return self.payload()

  def clear(self):
    self._snapshot()
    self.points = {}
    self.locations = []
    self.write()
    return self.payload()

  def preview_jpeg(self, name):
    if self.observe:
      if name not in self.observe["points"]:
        raise ValueError("observe point {} was not found".format(name))
      point = self.observe["points"][name]
      camera_files = {
        camera: find_observe_image(
          self.observe["image_root"], camera, point["source_frame"]
        )
        for camera in self.observe["cameras"]
        if camera in point["observations"]
      }
      point_annotations = point["observations"]
    elif self.marker_images and name in self.marker_images["captures"]:
      camera_files = self.marker_images["captures"][name]["files"]
      point_annotations = {}
    else:
      raise ValueError("image point {} was not found".format(name))
    cameras, images, annotations = [], [], {}
    for camera in self.cameras if self.marker_images else self.observe["cameras"]:
      filename = camera_files.get(camera)
      image = cv2.imread(str(filename)) if filename else None
      if image is None:
        continue
      cameras.append(camera)
      images.append(image)
      if camera in point_annotations:
        annotations[camera] = point_annotations[camera]
    if not images:
      raise ValueError("source images for {} were not found".format(name))
    mosaic, _placements = compose_mosaic(
      cameras, images, annotations,
      columns=min(2, len(cameras)), tile_width=560
    )
    ok, encoded = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
      raise ValueError("could not encode observe preview")
    return encoded.tobytes()


def make_handler(state):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
      return

    def _send(self, status, body, content_type):
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def _json(self, status, value):
      self._send(
        status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
        "application/json; charset=utf-8"
      )

    def do_GET(self):
      parsed = urlparse(self.path)
      try:
        if parsed.path == "/api/state":
          self._json(200, state.payload())
          return
        if parsed.path in ("/api/preview", "/api/observe"):
          name = parse_qs(parsed.query).get("point", [""])[0]
          self._send(200, state.preview_jpeg(name), "image/jpeg")
          return
        assets = {
          "/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/styles.css": ("styles.css", "text/css; charset=utf-8")
        }
        if parsed.path not in assets:
          self._json(404, {"error": "not found"})
          return
        filename, content_type = assets[parsed.path]
        self._send(200, (STATIC_ROOT / filename).read_bytes(), content_type)
      except (OSError, ValueError) as error:
        self._json(404, {"error": str(error)})

    def do_POST(self):
      try:
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/save":
          result = state.save(data)
        elif self.path == "/api/undo":
          result = state.undo()
        elif self.path == "/api/clear":
          result = state.clear()
        else:
          self._json(404, {"error": "not found"})
          return
        self._json(200, result)
      except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        self._json(400, {"error": str(error)})

  return Handler


def create_server(options):
  state = WorldpointsWebState(options)
  try:
    server = ThreadingHTTPServer(
      (options.host, options.port), make_handler(state)
    )
  except OSError as error:
    if error.errno != errno.EADDRINUSE or options.port == 0:
      raise
    server = ThreadingHTTPServer(
      (options.host, 0), make_handler(state)
    )
    print(
      "Worldpoints port {} is already in use; using port {} instead.".format(
        options.port, server.server_address[1]),
      flush=True,
    )
  server.state = state
  return server


def serve_worldpoints(options):
  server = create_server(options)
  host, port = server.server_address[:2]
  browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
  url = "http://{}:{}/".format(browser_host, port)
  print("World coordinate editor: {}".format(url), flush=True)
  print("Press Ctrl+C to stop", flush=True)
  if not options.no_browser:
    threading.Timer(.25, lambda: webbrowser.open(url)).start()
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
