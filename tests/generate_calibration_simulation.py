"""Generate synthetic multi-camera datasets for all Multical board types.

Supports:
  * checkerboard, ChArUco, and AprilGrid
  * combined intrinsic/extrinsic images or split datasets
  * plane, L-shaped, and triangular-prism coded targets
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from multical.board.aprilgrid import AprilGrid
from multical.board.charuco import CharucoBoard
from multical.board.checkerboard import Checkerboard


CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6"]
IMAGE_SIZE = (1280, 720)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OVERVIEW_FRAMES_PER_PAGE = 10
OVERVIEW_THUMBNAIL_SIZE = (320, 180)
OVERVIEW_CAPTION_HEIGHT = 28
OVERVIEW_TITLE_HEIGHT = 54


def normalize(vector):
  return vector / np.linalg.norm(vector)


def look_at(camera_position, target):
  """Return an OpenCV world-to-camera transform."""
  forward = normalize(target - camera_position)
  right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
  down = np.cross(forward, right)
  rotation = np.stack([right, down, forward])
  translation = -rotation @ camera_position
  return rotation, translation


def make_transform(rotation, translation):
  transform = np.eye(4)
  transform[:3, :3] = rotation
  transform[:3, 3] = translation
  return transform


def euler_rotation(x_degrees, y_degrees, z_degrees):
  x, y, z = np.deg2rad([x_degrees, y_degrees, z_degrees])
  rx = np.array([
    [1, 0, 0],
    [0, np.cos(x), -np.sin(x)],
    [0, np.sin(x), np.cos(x)]
  ])
  ry = np.array([
    [np.cos(y), 0, np.sin(y)],
    [0, 1, 0],
    [-np.sin(y), 0, np.cos(y)]
  ])
  rz = np.array([
    [np.cos(z), -np.sin(z), 0],
    [np.sin(z), np.cos(z), 0],
    [0, 0, 1]
  ])
  return rz @ ry @ rx


def project_points(points_world, rotation, translation, intrinsic):
  points_camera = (rotation @ points_world.T).T + translation
  pixels = (intrinsic @ points_camera.T).T
  pixels = pixels[:, :2] / pixels[:, 2:3]
  return pixels.astype(np.float32), points_camera[:, 2]


def board_surface_bounds(board):
  """Physical raster boundary in the board's local x/y coordinates."""
  if isinstance(board, Checkerboard):
    width, height = board.size
    square = board.square_length
    return (-square, -square, width * square, height * square)
  if isinstance(board, CharucoBoard):
    width, height = board.size
    return (
      0.0, 0.0,
      width * board.square_length,
      height * board.square_length
    )
  if isinstance(board, AprilGrid):
    width, height = board.size
    gap = board.tag_length * board.tag_spacing
    content_width = (
      width * board.tag_length + (width - 1) * gap
    )
    content_height = (
      height * board.tag_length + (height - 1) * gap
    )
    return (-gap, -gap, content_width + gap, content_height + gap)
  raise TypeError("unsupported board {}".format(type(board).__name__))


def create_boards(board_type, board_style):
  face_count = {"plane": 1, "l": 2, "triangle": 3}[board_style]
  if board_type == "checkerboard" and board_style != "plane":
    raise ValueError(
      "checkerboard supports only --board-style plane; coded faces are "
      "required to distinguish L-shaped and triangular targets"
    )

  boards = []
  for face_index in range(face_count):
    if board_type == "checkerboard":
      board = Checkerboard(size=(8, 5), square_length=0.12)
    elif board_type == "charuco":
      board = CharucoBoard(
        size=(10, 7),
        square_length=0.10,
        marker_length=0.075,
        aruco_dict="5X5_1000",
        aruco_offset=face_index * 100,
        min_rows=3,
        min_points=12
      )
    elif board_type == "aprilgrid":
      board = AprilGrid(
        size=(6, 4),
        tag_length=0.16,
        tag_spacing=0.30,
        start_id=face_index * 100,
        tag_family="t36h11",
        border_bits=2,
        min_rows=2,
        min_points=12
      )
    else:
      raise ValueError("unknown board type {}".format(board_type))
    boards.append(board)
  return boards


def write_board_config(path, board_type, boards):
  lines = ["boards:"]
  for face_index, board in enumerate(boards):
    lines.append("  face_{}:".format(face_index))
    lines.append("    _type_: {}".format(board_type))
    lines.append("    size: [{}, {}]".format(*board.size))
    if isinstance(board, Checkerboard):
      lines.append("    square_length: {:.8f}".format(board.square_length))
      lines.append("    use_sb: true")
    elif isinstance(board, CharucoBoard):
      lines.append("    square_length: {:.8f}".format(board.square_length))
      lines.append("    marker_length: {:.8f}".format(board.marker_length))
      lines.append("    aruco_dict: '{}'".format(board.aruco_dict))
      lines.append("    aruco_offset: {}".format(board.aruco_offset))
      lines.append("    min_rows: {}".format(board.min_rows))
      lines.append("    min_points: {}".format(board.min_points))
    elif isinstance(board, AprilGrid):
      lines.append("    tag_length: {:.8f}".format(board.tag_length))
      lines.append("    tag_spacing: {:.8f}".format(board.tag_spacing))
      lines.append("    start_id: {}".format(board.start_id))
      lines.append("    tag_family: '{}'".format(board.tag_family))
      lines.append("    border_bits: {}".format(board.border_bits))
      lines.append("    min_rows: {}".format(board.min_rows))
      lines.append("    min_points: {}".format(board.min_points))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def face_transforms(boards, board_style):
  """Return fixed board-local to target-rig transforms."""
  bounds = board_surface_bounds(boards[0])
  min_x, min_y, max_x, max_y = bounds
  face_width = max_x - min_x
  center_local = np.array([
    (min_x + max_x) / 2,
    (min_y + max_y) / 2,
    0.0
  ])
  vertical_down = np.array([0.0, 0.0, -1.0])

  if board_style == "plane":
    normals = [np.array([0.0, -1.0, 0.0])]
    centers = [np.zeros(3)]
  elif board_style == "l":
    normals = [
      np.array([0.0, -1.0, 0.0]),
      np.array([1.0, 0.0, 0.0])
    ]
    centers = []
    for normal in normals:
      horizontal = -np.cross(vertical_down, normal)
      centers.append(horizontal * face_width / 2)
  elif board_style == "triangle":
    normal_angles = np.deg2rad([-90.0, 30.0, 150.0])
    normals = [
      np.array([np.cos(angle), np.sin(angle), 0.0])
      for angle in normal_angles
    ]
    apothem = face_width / (2 * np.sqrt(3))
    centers = [normal * apothem for normal in normals]
  else:
    raise ValueError("unknown board style {}".format(board_style))

  transforms = []
  for normal, center in zip(normals, centers):
    # Raster coordinates use x-right/y-down. The printed front therefore
    # points opposite the right-handed x-cross-y normal.
    horizontal = normalize(-np.cross(vertical_down, normal))
    rotation = np.column_stack([horizontal, vertical_down, -normal])
    translation = center - rotation @ center_local
    transforms.append(make_transform(rotation, translation))
  return transforms


def make_cameras(board_style):
  image_width, image_height = IMAGE_SIZE
  intrinsic = np.array([
    [900.0, 0.0, image_width / 2],
    [0.0, 895.0, image_height / 2],
    [0.0, 0.0, 1.0]
  ])
  distortion = np.zeros(5)

  if board_style == "plane":
    positions = np.array([
      [-3.5, -5.5, 1.5],
      [-2.0, -5.0, 2.0],
      [-0.7, -4.7, 1.4],
      [0.7, -4.7, 2.1],
      [2.0, -5.0, 1.6],
      [3.5, -5.5, 2.0]
    ])
  else:
    angle_degrees = (
      [-140, -110, -80, -45, -10, 30]
      if board_style == "l"
      else [-150, -90, -30, 30, 90, 150]
    )
    angles = np.deg2rad(angle_degrees)
    heights = [1.5, 2.0, 1.4, 2.1, 1.6, 2.0]
    positions = np.array([
      [3.6 * np.cos(angle), 3.6 * np.sin(angle), height]
      for angle, height in zip(angles, heights)
    ])

  cameras = {}
  for name, position in zip(CAMERA_NAMES, positions):
    rotation, translation = look_at(
      position, np.array([0.0, 0.0, 1.0])
    )
    cameras[name] = {
      "K": intrinsic.tolist(),
      "dist": distortion.tolist(),
      "R_world_to_camera": rotation.tolist(),
      "T_world_to_camera": translation.tolist(),
      "position_world": position.tolist()
    }
  return cameras


def prepare_patterns(boards):
  patterns = []
  for board in boards:
    if isinstance(board, AprilGrid):
      pattern = board.draw(pixels_mm=3, margin_mm=0)
    else:
      pattern = board.draw(pixels_mm=3, margin=0)
    height, width = pattern.shape
    patterns.append({
      "image": pattern,
      "source_outer": np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0]
      ], dtype=np.float32),
      "bounds": board_surface_bounds(board)
    })
  return patterns


def surface_corners(bounds):
  min_x, min_y, max_x, max_y = bounds
  return np.array([
    [min_x, min_y, 0.0],
    [max_x, min_y, 0.0],
    [max_x, max_y, 0.0],
    [min_x, max_y, 0.0]
  ])


def blank_image(rng):
  width, height = IMAGE_SIZE
  background = rng.normal(
    rng.uniform(185, 225), 1.5, (height, width)
  )
  return np.clip(background, 0, 255).astype(np.uint8)


def render_face(image, pattern, destination):
  homography = cv2.getPerspectiveTransform(
    pattern["source_outer"], destination
  )
  rendered = cv2.warpPerspective(
    pattern["image"],
    homography,
    IMAGE_SIZE,
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=205
  )
  mask_source = np.full(pattern["image"].shape, 255, dtype=np.uint8)
  mask = cv2.warpPerspective(
    mask_source,
    homography,
    IMAGE_SIZE,
    flags=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
  )
  image[mask > 0] = rendered[mask > 0]


def finish_image(image, rng):
  image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.15, 0.75))
  noise = rng.normal(0, rng.uniform(0.35, 1.5), image.shape)
  return np.clip(image + noise, 0, 255).astype(np.uint8)


def visible_projection(
    face_world, pattern, camera, require_front=True, min_area=5000):
  rotation = np.asarray(camera["R_world_to_camera"])
  translation = np.asarray(camera["T_world_to_camera"])
  intrinsic = np.asarray(camera["K"])
  corners_local = surface_corners(pattern["bounds"])
  corners_world = (
    face_world[:3, :3] @ corners_local.T
  ).T + face_world[:3, 3]
  destination, depths = project_points(
    corners_world, rotation, translation, intrinsic
  )

  width, height = IMAGE_SIZE
  area = abs(cv2.contourArea(destination))
  within_image = (
    destination[:, 0].min() >= 4 and
    destination[:, 1].min() >= 4 and
    destination[:, 0].max() < width - 4 and
    destination[:, 1].max() < height - 4
  )
  in_front = np.all(depths > 0.1)

  if require_front:
    face_center = np.mean(corners_world, axis=0)
    printed_front_world = -face_world[:3, 2]
    to_camera = (
      np.asarray(camera["position_world"]) - face_center
    )
    front_facing = (
      np.dot(printed_front_world, normalize(to_camera)) > 0.08
    )
  else:
    front_facing = True

  valid = in_front and within_image and front_facing and area >= min_area
  return valid, destination


def save_image(path, image):
  cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])


def natural_sort_key(value):
  """Sort names containing numbers in human order (frame_2 before frame_10)."""
  return [
    (1, int(part)) if part.isdigit() else (0, part.lower())
    for part in re.split(r"(\d+)", str(value))
  ]


def overview_frame_rows(dataset_root):
  """Return frame names and camera paths in synchronized display order."""
  camera_files = {}
  frame_names = set()
  for camera_name in CAMERA_NAMES:
    files = {
      path.name: path
      for path in (dataset_root / camera_name).iterdir()
      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    camera_files[camera_name] = files
    frame_names.update(files)

  return [
    (
      frame_name,
      [camera_files[camera_name].get(frame_name)
       for camera_name in CAMERA_NAMES]
    )
    for frame_name in sorted(frame_names, key=natural_sort_key)
  ]


def resize_for_overview(image):
  """Fit an image inside one thumbnail cell without changing its aspect ratio."""
  target_width, target_height = OVERVIEW_THUMBNAIL_SIZE
  if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  source_height, source_width = image.shape[:2]
  scale = min(
    target_width / source_width,
    target_height / source_height
  )
  resized_width = max(1, round(source_width * scale))
  resized_height = max(1, round(source_height * scale))
  resized = cv2.resize(
    image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
  )
  thumbnail = np.full(
    (target_height, target_width, 3), 224, dtype=np.uint8
  )
  offset_x = (target_width - resized_width) // 2
  offset_y = (target_height - resized_height) // 2
  thumbnail[
    offset_y:offset_y + resized_height,
    offset_x:offset_x + resized_width
  ] = resized
  return thumbnail


def create_dataset_overviews(
    output_path, dataset_roots,
    frames_per_page=OVERVIEW_FRAMES_PER_PAGE):
  """Create paged contact sheets containing every generated camera image."""
  overview_directory = output_path / "overviews"
  overview_directory.mkdir(parents=True, exist_ok=True)
  overview_files = []
  total_image_count = 0
  sections = []
  thumbnail_width, thumbnail_height = OVERVIEW_THUMBNAIL_SIZE
  cell_height = thumbnail_height + OVERVIEW_CAPTION_HEIGHT

  for dataset_root in dataset_roots:
    section_name = (
      "combined" if dataset_root == output_path else dataset_root.name
    )
    rows = overview_frame_rows(dataset_root)
    image_count = sum(
      path is not None
      for _, camera_paths in rows
      for path in camera_paths
    )
    total_image_count += image_count
    page_count = max(
      1, (len(rows) + frames_per_page - 1) // frames_per_page
    )
    section_files = []

    for page_index in range(page_count):
      page_rows = rows[
        page_index * frames_per_page:
        (page_index + 1) * frames_per_page
      ]
      canvas_height = (
        OVERVIEW_TITLE_HEIGHT + max(1, len(page_rows)) * cell_height
      )
      canvas = np.full(
        (
          canvas_height,
          thumbnail_width * len(CAMERA_NAMES),
          3
        ),
        246,
        dtype=np.uint8
      )
      if page_rows:
        frame_range = "{} - {}".format(
          page_rows[0][0], page_rows[-1][0]
        )
      else:
        frame_range = "no images"
      title = "{} | page {}/{} | {}".format(
        section_name, page_index + 1, page_count, frame_range
      )
      cv2.putText(
        canvas, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (32, 32, 32), 2, cv2.LINE_AA
      )

      for row_index, (frame_name, camera_paths) in enumerate(page_rows):
        cell_y = OVERVIEW_TITLE_HEIGHT + row_index * cell_height
        for column_index, (camera_name, image_path) in enumerate(
            zip(CAMERA_NAMES, camera_paths)):
          cell_x = column_index * thumbnail_width
          if image_path is None:
            thumbnail = np.full(
              (thumbnail_height, thumbnail_width, 3),
              (210, 210, 245),
              dtype=np.uint8
            )
            cv2.putText(
              thumbnail, "missing", (58, 60),
              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 180),
              1, cv2.LINE_AA
            )
          else:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
              raise RuntimeError(
                "could not read generated image {}".format(image_path)
              )
            thumbnail = resize_for_overview(image)
          canvas[
            cell_y:cell_y + thumbnail_height,
            cell_x:cell_x + thumbnail_width
          ] = thumbnail
          cv2.rectangle(
            canvas,
            (cell_x, cell_y),
            (cell_x + thumbnail_width - 1, cell_y + cell_height - 1),
            (190, 190, 190),
            1
          )
          caption = "{}/{}".format(camera_name, frame_name)
          cv2.putText(
            canvas,
            caption,
            (cell_x + 7, cell_y + thumbnail_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (45, 45, 45),
            1,
            cv2.LINE_AA
          )

      overview_path = (
        overview_directory /
        "overview_{}_{:03d}.jpg".format(section_name, page_index + 1)
      )
      written = cv2.imwrite(
        str(overview_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]
      )
      if not written:
        raise RuntimeError(
          "could not write overview image {}".format(overview_path)
        )
      relative_path = str(overview_path.relative_to(output_path))
      overview_files.append(relative_path)
      section_files.append(relative_path)

    sections.append({
      "name": section_name,
      "frame_count": len(rows),
      "image_count": image_count,
      "files": section_files
    })

  return {
    "order": "dataset section, frame filename, cameras C1-C6",
    "frames_per_page": frames_per_page,
    "image_count": total_image_count,
    "files": overview_files,
    "sections": sections
  }


def random_rig_pose(rng, board_style):
  center = np.array([
    rng.uniform(-0.75, 0.75),
    rng.uniform(-0.60, 0.60),
    rng.uniform(0.75, 1.45)
  ])
  z_range = 32 if board_style == "plane" else 18
  rotation = euler_rotation(
    rng.uniform(-12, 12),
    rng.uniform(-12, 12),
    rng.uniform(-z_range, z_range)
  )
  return make_transform(rotation, center)


def generate_extrinsic_frames(
    output_path, frame_count, rng, cameras, boards, patterns,
    fixed_faces, board_style, filename_prefix="frame"):
  counts = {name: 0 for name in CAMERA_NAMES}
  rig_poses = []
  for frame_index in range(frame_count):
    rig_world = random_rig_pose(rng, board_style)
    rig_poses.append(rig_world.tolist())
    for camera_name in CAMERA_NAMES:
      image = blank_image(rng)
      visible_faces = 0
      for pattern, face_rig in zip(patterns, fixed_faces):
        face_world = rig_world @ face_rig
        visible, destination = visible_projection(
          face_world, pattern, cameras[camera_name], require_front=True
        )
        if visible:
          render_face(image, pattern, destination)
          visible_faces += 1
      if visible_faces:
        counts[camera_name] += 1
      save_image(
        output_path / camera_name /
        "{}_{:04d}.jpg".format(filename_prefix, frame_index),
        finish_image(image, rng)
      )
  return counts, rig_poses


def targeted_intrinsic_face(rng, pattern, camera):
  """Sample one planar target pose that covers varied image locations."""
  intrinsic = np.asarray(camera["K"])
  bounds = pattern["bounds"]
  corners = surface_corners(bounds)
  local_center = corners.mean(axis=0)
  width, height = IMAGE_SIZE

  for _ in range(200):
    depth = rng.uniform(2.0, 4.0)
    target_u = rng.uniform(width * 0.20, width * 0.80)
    target_v = rng.uniform(height * 0.18, height * 0.82)
    center_camera = np.array([
      (target_u - intrinsic[0, 2]) * depth / intrinsic[0, 0],
      (target_v - intrinsic[1, 2]) * depth / intrinsic[1, 1],
      depth
    ])
    rotation = euler_rotation(
      rng.uniform(-28, 28),
      rng.uniform(-35, 35),
      rng.uniform(-35, 35)
    )
    translation = center_camera - rotation @ local_center
    face_camera = make_transform(rotation, translation)

    destination, depths = project_points(
      (rotation @ corners.T).T + translation,
      np.eye(3), np.zeros(3), intrinsic
    )
    area = abs(cv2.contourArea(destination))
    valid = (
      np.all(depths > 0.1) and
      destination[:, 0].min() >= 5 and
      destination[:, 1].min() >= 5 and
      destination[:, 0].max() < width - 5 and
      destination[:, 1].max() < height - 5 and
      area >= 12000
    )
    if valid:
      return face_camera, destination
  raise RuntimeError("could not sample a visible intrinsic target pose")


def generate_split_intrinsic(
    output_path, frame_count, rng, cameras, patterns):
  counts = {}
  pattern = patterns[0]
  poses = {}
  for camera_name in CAMERA_NAMES:
    counts[camera_name] = 0
    poses[camera_name] = []
    for frame_index in range(frame_count):
      image = blank_image(rng)
      face_camera, destination = targeted_intrinsic_face(
        rng, pattern, cameras[camera_name]
      )
      render_face(image, pattern, destination)
      counts[camera_name] += 1
      poses[camera_name].append(face_camera.tolist())
      save_image(
        output_path / camera_name / "frame_{:04d}.jpg".format(frame_index),
        finish_image(image, rng)
      )
  return counts, poses


def generate_combined_intrinsic(
    output_path, frame_count, rng, cameras, patterns, start_index):
  """Add synchronized frames where only one camera sees a targeted board."""
  counts = {name: 0 for name in CAMERA_NAMES}
  pattern = patterns[0]
  metadata = []
  current_index = start_index
  for target_name in CAMERA_NAMES:
    for _ in range(frame_count):
      _, destination = targeted_intrinsic_face(
        rng, pattern, cameras[target_name]
      )
      filename = "frame_{:04d}.jpg".format(current_index)
      for camera_name in CAMERA_NAMES:
        image = blank_image(rng)
        if camera_name == target_name:
          render_face(image, pattern, destination)
          counts[camera_name] += 1
        save_image(
          output_path / camera_name / filename,
          finish_image(image, rng)
        )
      metadata.append({"frame": current_index, "camera": target_name})
      current_index += 1
  return counts, metadata


def ensure_output_directories(output_path, dataset_mode):
  if output_path.exists():
    existing_images = [
      path for path in output_path.rglob("*")
      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if existing_images:
      raise FileExistsError(
        "{} already contains generated images; choose a new --output-dir"
        .format(output_path)
      )
  output_path.mkdir(parents=True, exist_ok=True)

  roots = [output_path] if dataset_mode == "combined" else [
    output_path / "intrinsic", output_path / "extrinsic"
  ]
  for root in roots:
    for camera_name in CAMERA_NAMES:
      (root / camera_name).mkdir(parents=True, exist_ok=True)
  return roots


def relative_camera_poses(cameras):
  reference = make_transform(
    np.asarray(cameras["C1"]["R_world_to_camera"]),
    np.asarray(cameras["C1"]["T_world_to_camera"])
  )
  poses = {}
  for name in CAMERA_NAMES:
    camera_transform = make_transform(
      np.asarray(cameras[name]["R_world_to_camera"]),
      np.asarray(cameras[name]["T_world_to_camera"])
    )
    relative = camera_transform @ np.linalg.inv(reference)
    poses[name] = {
      "R": relative[:3, :3].tolist(),
      "T": relative[:3, 3].tolist()
    }
  return poses


def board_truth(board):
  data = dict(board.export())
  for key, value in list(data.items()):
    if isinstance(value, tuple):
      data[key] = list(value)
  return data


def write_commands(path, dataset_mode, intrinsic_frames):
  cameras = " ".join(CAMERA_NAMES)
  output = str(path)
  if dataset_mode == "combined":
    commands = [
      "uv run multical calibrate \\",
      "  --image_path {} \\".format(output),
      "  --boards {}/boards.yaml \\".format(output),
      "  --cameras {} \\".format(cameras),
      "  --master C1 --loss soft_l1 --iter 3",
      "",
      "uv run python tests/evaluate_calibration_simulation.py \\",
      "  --calibration {}/calibration.json \\".format(output),
      "  --ground-truth {}/ground_truth.json".format(output)
    ]
  else:
    commands = [
      "uv run multical intrinsic \\",
      "  --image_path {}/intrinsic \\".format(output),
      "  --boards {}/boards.yaml \\".format(output),
      "  --cameras {} --limit_intrinsic {}".format(
        cameras, intrinsic_frames
      ),
      "",
      "uv run multical calibrate \\",
      "  --image_path {}/extrinsic \\".format(output),
      "  --boards {}/boards.yaml \\".format(output),
      "  --cameras {} \\".format(cameras),
      "  --calibration {}/intrinsic/intrinsic.json \\".format(output),
      "  --fix_intrinsic --master C1 --loss soft_l1 --iter 3",
      "",
      "uv run python tests/evaluate_calibration_simulation.py \\",
      "  --calibration {}/extrinsic/calibration.json \\".format(output),
      "  --ground-truth {}/ground_truth.json".format(output)
    ]
  (path / "commands.txt").write_text(
    "\n".join(commands) + "\n", encoding="utf-8"
  )


def generate(
    output_dir, board_type="checkerboard", board_style="plane",
    dataset_mode="combined", intrinsic_frames=20, extrinsic_frames=50,
    seed=17):
  output_path = Path(output_dir).resolve()
  roots = ensure_output_directories(output_path, dataset_mode)
  rng = np.random.default_rng(seed)
  boards = create_boards(board_type, board_style)
  patterns = prepare_patterns(boards)
  fixed_faces = face_transforms(boards, board_style)
  cameras = make_cameras(board_style)

  write_board_config(output_path / "boards.yaml", board_type, boards)
  if dataset_mode == "split":
    write_board_config(
      output_path / "intrinsic" / "boards.yaml", board_type, boards
    )
    write_board_config(
      output_path / "extrinsic" / "boards.yaml", board_type, boards
    )

  if dataset_mode == "combined":
    root = roots[0]
    extrinsic_counts, rig_poses = generate_extrinsic_frames(
      root, extrinsic_frames, rng, cameras, boards, patterns, fixed_faces,
      board_style
    )
    intrinsic_counts, intrinsic_metadata = generate_combined_intrinsic(
      root, intrinsic_frames, rng, cameras, patterns, extrinsic_frames
    )
    dataset = {
      "mode": "combined",
      "path": str(root),
      "extrinsic_frames": extrinsic_frames,
      "intrinsic_frames_per_camera": intrinsic_frames,
      "total_synchronized_frames": (
        extrinsic_frames + intrinsic_frames * len(CAMERA_NAMES)
      ),
      "extrinsic_visible_frames": extrinsic_counts,
      "targeted_intrinsic_frames": intrinsic_counts,
      "targeted_intrinsic_metadata": intrinsic_metadata,
      "rig_poses": rig_poses
    }
  else:
    intrinsic_root, extrinsic_root = roots
    intrinsic_counts, intrinsic_poses = generate_split_intrinsic(
      intrinsic_root, intrinsic_frames, rng, cameras, patterns
    )
    extrinsic_counts, rig_poses = generate_extrinsic_frames(
      extrinsic_root, extrinsic_frames, rng, cameras, boards,
      patterns, fixed_faces, board_style
    )
    dataset = {
      "mode": "split",
      "intrinsic_path": str(intrinsic_root),
      "extrinsic_path": str(extrinsic_root),
      "intrinsic_frames_per_camera": intrinsic_frames,
      "extrinsic_frames": extrinsic_frames,
      "intrinsic_visible_frames": intrinsic_counts,
      "extrinsic_visible_frames": extrinsic_counts,
      "intrinsic_poses_camera_frame": intrinsic_poses,
      "rig_poses": rig_poses
    }

  overview = create_dataset_overviews(output_path, roots)
  dataset["overview"] = overview
  truth = {
    "image_size": list(IMAGE_SIZE),
    "board_type": board_type,
    "board_style": board_style,
    "boards": [board_truth(board) for board in boards],
    "board_poses_in_rig": [pose.tolist() for pose in fixed_faces],
    "cameras": cameras,
    "camera_poses_relative_to_C1": relative_camera_poses(cameras),
    "dataset": dataset,
    "seed": seed
  }
  (output_path / "ground_truth.json").write_text(
    json.dumps(truth, indent=2), encoding="utf-8"
  )
  write_commands(output_path, dataset_mode, intrinsic_frames)

  return {
    "output_dir": str(output_path),
    "board_type": board_type,
    "board_style": board_style,
    "dataset_mode": dataset_mode,
    "intrinsic_frames_per_camera": intrinsic_frames,
    "extrinsic_frames": extrinsic_frames,
    "commands": str(output_path / "commands.txt"),
    "overview_files": [
      str(output_path / relative_path)
      for relative_path in overview["files"]
    ],
    "overview_image_count": overview["image_count"]
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", default="calibration_simulation_v2")
  parser.add_argument(
    "--board-type",
    choices=["checkerboard", "charuco", "aprilgrid"],
    default="checkerboard"
  )
  parser.add_argument(
    "--board-style",
    choices=["plane", "l", "triangle"],
    default="plane"
  )
  parser.add_argument(
    "--dataset-mode",
    choices=["combined", "split"],
    default="combined"
  )
  parser.add_argument("--intrinsic-frames", type=int, default=20)
  parser.add_argument("--extrinsic-frames", type=int, default=50)
  parser.add_argument("--seed", type=int, default=17)
  arguments = parser.parse_args()
  print(json.dumps(generate(
    output_dir=arguments.output_dir,
    board_type=arguments.board_type,
    board_style=arguments.board_style,
    dataset_mode=arguments.dataset_mode,
    intrinsic_frames=arguments.intrinsic_frames,
    extrinsic_frames=arguments.extrinsic_frames,
    seed=arguments.seed
  ), indent=2))
