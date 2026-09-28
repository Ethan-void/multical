from pathlib import Path
from types import SimpleNamespace
import json
import socket
import threading
from urllib.request import Request, urlopen

import matplotlib
import cv2
import numpy as np
import pytest
import yaml

from multical.app.worldpoints import (
  measured_world_points_document,
  find_observe_image,
  load_observe_points,
  load_marker_board,
  parse_marker_specs,
  world_markers_document,
  write_world_config,
  WorldPointEditor,
  Worldpoints,
)
from multical.app.worldpoints_web import (
  WorldpointsWebState,
  create_server,
  load_marker_image_points,
)


matplotlib.use("Agg", force=True)


def test_measured_world_points_document_and_yaml(tmp_path):
  document = measured_world_points_document([
    ("P01", (0, 4.115, 0)),
    ("P02", (14.251, -0.2, 1.895)),
  ])
  destination = write_world_config(
    tmp_path / "measured_world_points.yaml", document
  )
  loaded = yaml.safe_load(destination.read_text(encoding="utf-8"))
  assert loaded == {
    "coordinate_frame": "world",
    "world_units": "meters",
    "points": {
      "P01": [0.0, 4.115, 0.0],
      "P02": [14.251, -0.2, 1.895],
    },
  }
  assert "P01: [0.0, 4.115, 0.0]" in destination.read_text()


def test_world_markers_multiple_fixed_heights(tmp_path):
  specs = parse_marker_specs("12,10", "1.690,0.590", "upper,lower")
  document = world_markers_document(
    [("000000", 23.77, 2.0)], ["cam0", "cam1"], specs
  )
  destination = write_world_config(tmp_path / "world_markers.yaml", document)
  loaded = yaml.safe_load(Path(destination).read_text(encoding="utf-8"))
  assert loaded["cameras"] == ["cam0", "cam1"]
  assert loaded["captures"][0] == {
    "name": "000000",
    "markers": [
      {
        "marker_id": 12,
        "occurrence": "upper",
        "world_point": [23.77, 2.0, 1.69],
      },
      {
        "marker_id": 10,
        "occurrence": "lower",
        "world_point": [23.77, 2.0, 0.59],
      },
    ],
  }


def test_marker_specs_allow_no_occurrence_and_validate_lengths():
  assert parse_marker_specs("23", "1.7") == [
    {"marker_id": 23, "height": 1.7}
  ]
  with pytest.raises(ValueError, match="equal counts"):
    parse_marker_specs("12,10", "1.69")


def test_marker_board_yaml_loads_reusable_marker_specs(tmp_path):
  board = tmp_path / "world_board.yaml"
  board.write_text(
    "marker_family: 6X6_250\n"
    "markers:\n"
    "  - marker_id: 12\n"
    "    height: 1.690\n"
    "  - marker_id: 10\n"
    "    height: 0.590\n",
    encoding="utf-8",
  )

  family, specs = load_marker_board(board)

  assert family == "6X6_250"
  assert specs == [
    {"marker_id": 12, "height": 1.69},
    {"marker_id": 10, "height": 0.59},
  ]


def test_measured_point_is_committed_only_by_add_button(tmp_path):
  editor = WorldPointEditor(Worldpoints(
    mode="measured",
    output=str(tmp_path / "measured_world_points.yaml"),
  ))
  editor._on_click(SimpleNamespace(
    button=1, xdata=5.485, ydata=4.115, inaxes=editor.axtop
  ))
  assert editor.pending_xy is True
  assert editor.points == []
  assert editor.name_box.text == "P01"

  editor._on_click(SimpleNamespace(
    button=1, xdata=12.0, ydata=1.45, inaxes=editor.axside
  ))
  assert editor.pending_xy is True
  assert editor.points == []
  assert editor.name_box.text == "P01"
  assert not (tmp_path / "measured_world_points.yaml").exists()

  editor._add_or_update(None)
  assert editor.pending_xy is False
  assert editor.points == [("P01", (5.485, 4.115, 1.45))]
  assert editor.name_box.text == "P02"
  loaded = yaml.safe_load(
    (tmp_path / "measured_world_points.yaml").read_text(encoding="utf-8")
  )
  assert loaded["points"]["P01"] == [5.485, 4.115, 1.45]


def test_observe_file_drives_clickable_point_list_without_being_modified(
    tmp_path):
  image_root = tmp_path / "images"
  for camera in ("cam0", "cam1"):
    directory = image_root / camera
    directory.mkdir(parents=True)
    cv2.imwrite(
      str(directory / "000000.jpg"),
      np.full((120, 160, 3), 180, dtype=np.uint8)
    )
  observe_path = tmp_path / "measured_observations.yaml"
  observe_path.write_text(yaml.safe_dump({
    "source": {"image_path": str(image_root), "mode": "manual_click"},
    "cameras": ["cam0", "cam1"],
    "frames": [
      {
        "frame": "P01",
        "source_frame": "000000.jpg",
        "observations": {"cam0": [25, 35], "cam1": [30, 40]},
      },
      {
        "frame": "P02",
        "source_frame": "000000.jpg",
        "observations": {"cam0": [50, 60], "cam1": [55, 65]},
      },
    ],
  }, sort_keys=False), encoding="utf-8")
  original_observe = observe_path.read_bytes()

  loaded = load_observe_points(observe_path)
  assert list(loaded["points"]) == ["P01", "P02"]
  assert loaded["selection_mode"] == "single_image_multiple_points"
  assert loaded["points"]["P01"]["observations"]["cam0"] == [25.0, 35.0]
  assert find_observe_image(
    loaded["image_root"], "cam1", "000000.jpg"
  ) == image_root / "cam1" / "000000.jpg"

  output = tmp_path / "measured_world_points.yaml"
  editor = WorldPointEditor(Worldpoints(
    mode="measured", output=str(output), observe=str(observe_path)
  ))
  editor._show_observe_result("P01")
  assert editor.observation_figure is not None
  assert editor.observation_figure.canvas.manager.get_window_title() == (
    "Observe result - P01"
  )
  shown = []
  editor._show_observe_result = shown.append
  editor._on_click(SimpleNamespace(
    button=1, xdata=.5, ydata=.72, inaxes=editor.summary_axis
  ))
  assert editor.active_observation == "P01"
  assert shown == ["P01"]
  editor.x_box.set_val("5.485")
  editor.y_box.set_val("4.115")
  editor.z_box.set_val("1.450")
  editor._add_or_update(None)
  assert editor.active_observation == "P02"
  assert yaml.safe_load(output.read_text())["points"]["P01"] == [
    5.485, 4.115, 1.45
  ]
  assert observe_path.read_bytes() == original_observe


def test_copied_observe_dataset_prefers_colocated_images(tmp_path):
  original_images = tmp_path / "original" / "measured_points"
  original_images.mkdir(parents=True)
  copied_observe = tmp_path / "copied" / "observe"
  copied_images = copied_observe / "measured_points"
  copied_images.mkdir(parents=True)
  observe_path = copied_observe / "measured_observations.yaml"
  observe_path.write_text(yaml.safe_dump({
    "source": {"image_path": str(original_images)},
    "cameras": ["cam0"],
    "frames": [{
      "frame": "P01",
      "source_frame": "000000.jpg",
      "observations": {"cam0": [10, 20]},
    }],
  }), encoding="utf-8")

  loaded = load_observe_points(observe_path)
  assert loaded["image_root"] == copied_images.resolve()


def test_multiple_images_single_point_mode_uses_frame_names(tmp_path):
  observe_path = tmp_path / "measured_observations.yaml"
  observe_path.write_text(yaml.safe_dump({
    "source": {"image_path": str(tmp_path / "measured_points")},
    "cameras": ["cam0"],
    "frames": [
      {"frame": "000000.jpg", "observations": {"cam0": [10, 20]}},
      {"frame": "000001.jpg", "observations": {"cam0": [11, 21]}},
    ],
  }), encoding="utf-8")

  loaded = load_observe_points(observe_path)
  assert loaded["selection_mode"] == "multiple_images_single_point"
  assert list(loaded["points"]) == ["000000.jpg", "000001.jpg"]
  assert loaded["points"]["000001.jpg"]["source_frame"] == "000001.jpg"


def test_worldpoints_web_serves_observe_and_saves_only_output(tmp_path):
  image_root = tmp_path / "observe" / "measured_points" / "cam0"
  image_root.mkdir(parents=True)
  cv2.imwrite(
    str(image_root / "000000.jpg"),
    np.full((120, 160, 3), 180, dtype=np.uint8)
  )
  observe_path = tmp_path / "observe" / "measured_observations.yaml"
  observe_path.write_text(yaml.safe_dump({
    "source": {"image_path": str(image_root.parent)},
    "cameras": ["cam0"],
    "frames": [{
      "frame": "P01", "source_frame": "000000.jpg",
      "observations": {"cam0": [30, 40]},
    }],
  }), encoding="utf-8")
  original_observe = observe_path.read_bytes()
  output = tmp_path / "measured_world_points.yaml"
  options = Worldpoints(
    mode="measured", observe=str(observe_path), output=str(output),
    host="127.0.0.1", port=0, no_browser=True
  )
  server = create_server(options)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  base = "http://127.0.0.1:{}".format(server.server_address[1])
  try:
    assert "世界坐标编辑器" in urlopen(base + "/").read().decode("utf-8")
    state = json.loads(urlopen(base + "/api/state").read())
    assert state["point_names"] == ["P01"]
    preview = urlopen(base + "/api/observe?point=P01")
    assert preview.headers.get_content_type() == "image/jpeg"
    assert preview.read(2) == b"\xff\xd8"
    payload = json.dumps({
      "name": "P01", "x": 5.485, "y": 4.115, "z": 1.45
    }).encode("utf-8")
    response = urlopen(Request(
      base + "/api/save", data=payload,
      headers={"Content-Type": "application/json"}, method="POST"
    ))
    saved = json.loads(response.read())
    assert saved["points"]["P01"] == [5.485, 4.115, 1.45]
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
  assert yaml.safe_load(output.read_text())["points"]["P01"] == [
    5.485, 4.115, 1.45
  ]
  assert observe_path.read_bytes() == original_observe


def test_worldpoints_web_uses_free_port_when_requested_port_is_busy(tmp_path):
  blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  blocker.bind(("127.0.0.1", 0))
  blocker.listen(1)
  busy_port = blocker.getsockname()[1]
  options = Worldpoints(
    mode="measured",
    output=str(tmp_path / "measured_world_points.yaml"),
    host="127.0.0.1",
    port=busy_port,
    no_browser=True,
  )
  server = None
  try:
    server = create_server(options)
    assert server.server_address[1] != busy_port
    assert server.server_address[1] > 0
  finally:
    if server is not None:
      server.server_close()
    blocker.close()


def test_web_top_view_contains_dimensions_and_snap_anchors():
  web_root = Path(__file__).parents[1] / "multical" / "web" / "worldpoints"
  html = (web_root / "index.html").read_text(encoding="utf-8")
  styles = (web_root / "styles.css").read_text(encoding="utf-8")
  script = (web_root / "app.js").read_text(encoding="utf-8")
  main_column = html.split(
    '<section class="main-column">', 1
  )[1].split("</section>", 1)[0]
  control_column = html.split(
    '<aside class="control-column">', 1
  )[1].split("</aside>", 1)[0]
  assert 'id="topView"' in main_column
  assert 'id="observeImage"' in main_column
  assert 'id="court3d"' in main_column
  assert 'id="reset3d"' in main_column
  assert 'id="fullscreen3d"' in main_column
  assert 'id="fullscreenTop"' in main_column
  assert 'id="exportTop"' in main_column
  assert 'id="sideView"' not in html
  assert "drawSide" not in script
  assert 'id="pointList"' not in main_column
  assert control_column.index('class="panel editor-card"') < (
    control_column.index('id="pointList"')
  )
  assert "minmax(520px, 1fr) 290px" in styles
  assert ".control-column { position: sticky" in styles
  assert ".point-panel { height: clamp(" in styles
  assert "overflow-y: auto" in styles
  assert "min-height: calc(100vh - 116px)" in styles
  assert ".top-card #topView { display: block; width: 100%; height: 520px" in styles
  assert "#court3d { display: block; width: 100%; height: 460px" in styles
  assert ".top-card:fullscreen, .court-card:fullscreen" in styles
  assert "toggleFullscreen(elements.topCard)" in script
  assert "toggleFullscreen(elements.courtCard)" in script
  assert "event.altKey || document.fullscreenElement === elements.courtCard" in script
  assert "if (!zoomRequested) return" in script
  assert "Alt + 滚轮缩放" in html
  assert "function drawTop(exportCoordinates = false)" in script
  assert "elements.top.toBlob(" in script
  assert "`${point.name} ${coordinateText(point.name)}`" in script
  assert 'if (activeName && !exportCoordinates)' in script
  assert 'link.download = filename' in script
  assert 'document.addEventListener("fullscreenchange"' in script
  assert html.index('id="topView"') < html.index('id="observeImage"')
  assert html.index('id="observeImage"') < html.index('id="court3d"')
  assert "COURT = state.court" in script
  assert "COURT.lines.forEach" in script
  assert "x.toFixed(3)" in script
  assert "y.toFixed(3)" in script
  assert 'fillText("X =' not in script
  assert 'fillText("Y =' not in script
  assert "O (0, 0)" not in script
  assert 'ctx.fillStyle="#ff705c"' not in script
  assert "6.400" not in script
  assert "1.37 m" not in script
  assert "courtAnchors()" in script
  assert "nearest.distance <= 16" in script
  assert "anchor.inner?2.5:1.5" in script
  assert "const netTop" in script
  assert "const drawDimensionScale" in script
  assert '[[COURT.service,-6.25,.02],"5.485"]' not in script
  assert '[[COURT.length-COURT.service,-6.25,.02],"18.285"]' not in script
  assert '[[-1.45,-COURT.singles/2,.02],"-4.115"]' not in script
  assert '[[-1.45,COURT.singles/2,.02],"+4.115"]' not in script
  assert 'fillPolygon(runoff,"#b86142")' in script
  assert "DEFAULT_VIEW" in script
  assert "dy = -y" in script
  assert "const axisLabel" not in script
  assert 'class="axis-legend"' not in html
  assert "state?.snap_anchors" in script
  assert "configured:true" in script
  assert "function pointLabels()" in script
  assert "function drawPointLabel(" in script
  assert '"800 12px ui-monospace' in script
  assert '"800 10px ui-monospace' in script
  assert '"700 9px ui-monospace' in script
  assert "exportCoordinates ? .8 : 1.45" in script
  assert "function drawExportPointLabels(" in script
  assert "boxes.findIndex((box) => !overlaps(box))" in script
  assert "if (!exportCoordinates)" in script
  assert "function drawPoint2d(ctx, x, y, active, scale = 1)" in script
  assert "1.45" in script
  assert "ctx.strokeText(label" not in script
  assert 'ctx.shadowColor = "rgba(9,28,20,.72)"' in script
  assert "const size = 10" in script
  assert 'ctx.strokeStyle = "rgba(255,255,255,.95)"' in script
  assert "function coordinateText(name)" in script
  assert 'coordinate.className = "point-coordinate"' in script
  assert ".point-coordinate { color: var(--muted); font: 600 11px" in styles


def test_marker_image_groups_drive_capture_list_and_two_heights(tmp_path):
  image_root = tmp_path / "world" / "world_images"
  for camera in ("cam0", "cam1", "cam4", "cam5"):
    directory = image_root / camera
    directory.mkdir(parents=True)
    for frame in ("000000.jpg", "000001.jpg"):
      cv2.imwrite(
        str(directory / frame),
        np.full((80, 120, 3), 160, dtype=np.uint8)
      )

  output_01 = tmp_path / "world" / "world_markers_01.yaml"
  group_01 = load_marker_image_points(output_01, "world_images")
  assert group_01["cameras"] == ["cam0", "cam1"]
  assert list(group_01["captures"]) == ["000000", "000001"]
  assert group_01["config_image_path"] == "world_images"

  output_45 = tmp_path / "world" / "world_markers_45.yaml"
  assert load_marker_image_points(output_45, "world_images")["cameras"] == [
    "cam4", "cam5"
  ]
  generic = load_marker_image_points(
    tmp_path / "world" / "world_markers.yaml", "world_images"
  )
  assert generic["cameras"] == ["cam0", "cam1", "cam4", "cam5"]

  state = WorldpointsWebState(Worldpoints(
    mode="markers", output=str(output_01), image_path="world_images"
  ))
  assert state.point_names() == ["000000", "000001"]
  assert state.marker_specs == [
    {"marker_id": 23, "height": 1.7, "occurrence": "upper"},
    {"marker_id": 23, "height": 0.5, "occurrence": "lower"},
  ]
  assert state.preview_jpeg("000000")[:2] == b"\xff\xd8"
  state.save({
    "name": "000000", "x": 23.77, "y": 2.0,
    "marker_ids": "23,23", "marker_heights": "1.700,0.500",
    "marker_occurrences": "upper,lower",
  })
  written = yaml.safe_load(output_01.read_text(encoding="utf-8"))
  assert written["cameras"] == ["cam0", "cam1"]
  assert written["image_path"] == "world_images"
  assert [
    marker["world_point"][2]
    for marker in written["captures"][0]["markers"]
  ] == [1.7, 0.5]
  reloaded = WorldpointsWebState(Worldpoints(
    mode="markers", output=str(output_01), image_path="world_images"
  ))
  assert reloaded.payload()["snap_anchors"] == [{"x": 23.77, "y": 2.0}]
  reloaded.clear()
  assert reloaded.payload()["snap_anchors"] == [{"x": 23.77, "y": 2.0}]


def test_marker_yaml_only_fallback_without_image_directory(tmp_path):
  output = tmp_path / "world" / "world_markers_01.yaml"
  write_world_config(output, world_markers_document(
    [("000000", 23.77, 2.0)], ["cam0", "cam1"],
    [
      {"marker_id": 12, "height": 1.69},
      {"marker_id": 10, "height": 0.59},
    ], marker_family="6X6_250", image_path="world_images"
  ))

  state = WorldpointsWebState(Worldpoints(
    mode="markers", output=str(output), image_path="world_images"
  ))
  payload = state.payload()
  assert payload["image_mode"] == "marker_yaml"
  assert payload["has_images"] is False
  assert payload["image_warning"]
  assert payload["point_names"] == ["000000"]
  assert payload["cameras"] == ["cam0", "cam1"]
  assert payload["marker_specs"] == [
    {"marker_id": 12, "height": 1.69},
    {"marker_id": 10, "height": 0.59},
  ]
  state.save({
    "name": "000000", "x": 18.285, "y": 4.115,
    "marker_ids": "12,10", "marker_heights": "1.690,0.590",
    "marker_occurrences": "",
  })
  written = yaml.safe_load(output.read_text(encoding="utf-8"))
  assert written["cameras"] == ["cam0", "cam1"]
  assert written["image_path"] == "world_images"
  assert written["captures"][0]["markers"][0]["world_point"] == [
    18.285, 4.115, 1.69
  ]


def test_measured_yaml_only_fallback_without_observe(tmp_path):
  output = tmp_path / "measured_world_points.yaml"
  write_world_config(output, measured_world_points_document([
    ("P01", [5.485, 4.115, 1.45]),
    ("P02", [18.285, -4.115, 0.0]),
  ]))

  state = WorldpointsWebState(Worldpoints(
    mode="measured",
    observe=str(tmp_path / "missing" / "measured_observations.yaml"),
    output=str(output),
  ))
  payload = state.payload()
  assert payload["image_mode"] == "measured_yaml"
  assert payload["has_images"] is False
  assert payload["image_warning"]
  assert payload["point_names"] == ["P01", "P02", "P03"]
  assert payload["points"]["P01"] == [5.485, 4.115, 1.45]

  state.save({"name": "P01", "x": 0, "y": 0, "z": 2})
  written = yaml.safe_load(output.read_text(encoding="utf-8"))
  assert written["points"]["P01"] == [0.0, 0.0, 2.0]


@pytest.mark.parametrize("court,length,width", [
  ("tennis", 23.77, 10.97), ("badminton", 13.36, 6.06)
])
def test_court_geometry_reaches_web_and_desktop(tmp_path, court, length, width):
  import matplotlib.pyplot as plt
  from multical.app.worldpoints import court_geometry, _draw_court_2d, _draw_court_3d

  state = WorldpointsWebState(Worldpoints(
    court=court, output=str(tmp_path / "points.yaml")))
  geometry = state.payload()["court"]
  assert geometry == court_geometry(court)
  assert geometry["length"] == length
  assert geometry["doubles"] == width
  figure = plt.figure()
  try:
    top = figure.add_subplot(121)
    spatial = figure.add_subplot(122, projection="3d")
    _draw_court_2d(top, court)
    _draw_court_3d(spatial, 0, court)
    figure.canvas.draw()
    assert top.get_xlim() == pytest.approx((-.8, length+.8))
    assert spatial.get_ylim() == pytest.approx((-width/2, width/2))
    for line, expected in zip(top.lines, geometry["lines"]):
      assert list(line.get_xdata()) == [expected[0], expected[2]]
      assert list(line.get_ydata()) == [expected[1], expected[3]]
  finally:
    plt.close(figure)


def test_badminton_service_lines_and_net():
  from multical.app.worldpoints import court_geometry
  geometry = court_geometry("badminton")
  lines = geometry["lines"]
  assert [.76, -3.03, .76, 3.03] in lines
  assert [4.68, -3.03, 4.68, 3.03] in lines
  assert [0, 0, 4.68, 0] in lines
  assert [4.68, 0, 8.68, 0] not in lines
  assert all(not (a == c == 6.66) for a, b, c, d in lines)
  assert geometry["net_center"] == 1.524
  assert geometry["net_post"] == 1.55
  with pytest.raises(ValueError, match="court"):
    court_geometry("unknown")


def test_web_badminton_snap_intersections():
  import shutil
  import subprocess
  from multical.app.worldpoints import court_geometry
  node = shutil.which("node")
  if not node:
    pytest.skip("node unavailable")
  script = (Path(__file__).parents[1] / "multical/web/worldpoints/app.js").read_text()
  function = script.split("function courtAnchors() {", 1)[1].split("function drawTop", 1)[0]
  result = subprocess.run([node, "-e",
    "const COURT = " + json.dumps(court_geometry("badminton")) + ";"
    "const state = {snap_anchors:[{x:2.123,y:1.456}]};"
    "function courtAnchors() {" + function +
    "console.log(JSON.stringify(courtAnchors()));"],
    check=True, capture_output=True, text=True)
  anchors = {(round(p["x"], 3), round(p["y"], 3)) for p in json.loads(result.stdout)}
  for x in (.76, 4.68, 8.68, 12.6):
    for y in (-3.03, -2.57, 0, 2.57, 3.03):
      assert (x, y) in anchors
  assert (2.123, 1.456) in anchors
  for point in ((10.005, 3.03), (10.01, 1.48), (10.02, 0.),
                (10.05, -3.03), (10.03, -1.54), (12.03, 0.),
                (12.02, 1.48), (12.03, -1.54)):
    assert point in anchors
  assert (5.485, 4.115) not in anchors


def test_badminton_fixed_anchors_without_marker_files(tmp_path):
  state = WorldpointsWebState(Worldpoints(
    court="badminton", output=str(tmp_path / "measured.yaml")))
  anchors = state.payload()["court"]["snap_anchors"]
  assert len(anchors) == 13
  assert {"x": 10.03, "y": -1.54} in anchors
  assert {"x": 12.02, "y": 1.48} in anchors
  state.clear()
  assert state.payload()["court"]["snap_anchors"] == anchors
  from multical.app.worldpoints import court_geometry
  assert len(court_geometry("tennis")["snap_anchors"]) == 4


def test_tennis_fixed_anchors_without_marker_files(tmp_path):
  state = WorldpointsWebState(Worldpoints(
    court="tennis", output=str(tmp_path / "measured.yaml")))
  expected = [
    {"x": 12.485, "y": 4.115}, {"x": 13.485, "y": 0.},
    {"x": 12.485, "y": -4.115}, {"x": 10.485, "y": 4.115},
  ]
  assert state.payload()["court"]["snap_anchors"] == expected
  state.clear()
  assert state.payload()["court"]["snap_anchors"] == expected


def test_fixed_anchor_config_edits_and_empty_list(tmp_path, monkeypatch):
  import multical.app.worldpoints as module
  config = tmp_path / "anchors.yaml"
  monkeypatch.setattr(module, "SNAP_ANCHORS_CONFIG", config)
  config.write_text("tennis: [[1.25, -2.5], [1.25, -2.5]]\nbadminton: []\n")
  original_lines = module.court_geometry("tennis")["lines"]
  assert module.court_geometry("tennis")["snap_anchors"] == [{"x": 1.25, "y": -2.5}]
  assert module.court_geometry("badminton")["snap_anchors"] == []
  config.write_text("tennis: [[3, 4]]\n")
  assert module.court_geometry("tennis")["snap_anchors"] == [{"x": 3., "y": 4.}]
  assert module.court_geometry("tennis")["lines"] == original_lines


@pytest.mark.parametrize("contents", [
  "[]", "tennis: null", "tennis: [[1, 2, 3]]", "tennis: [[.nan, 2]]",
  "tennis: [[true, 2]]", "tennis: [[oops, 2]]", "tennis: [",
])
def test_fixed_anchor_config_errors_identify_file(tmp_path, monkeypatch, contents):
  import multical.app.worldpoints as module
  config = tmp_path / "anchors.yaml"
  monkeypatch.setattr(module, "SNAP_ANCHORS_CONFIG", config)
  config.write_text(contents)
  with pytest.raises(ValueError, match="anchors.yaml"):
    module.court_geometry("tennis")
