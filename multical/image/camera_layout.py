"""Export a metric court top view from world-to-camera extrinsics.

Chart contract: spatial camera comparison; one point per calibrated camera,
XY in meters with equal scale, XYZ labels, projected optical-axis arrows.
Reference court follows worldpoints.py: left baseline center is (0, 0, 0).
PNG export; explicit reference-inspired colors and direct camera labels.
"""
import json
from pathlib import Path

import numpy as np


def camera_poses(data):
  if data.get("world_units", "meters") != "meters":
    raise ValueError("camera layout requires world_units=meters")
  cameras = []
  for name, camera in sorted(data["cameras"].items()):
    pose = camera["world_to_camera"]
    rotation = np.asarray(pose["R"], dtype=float)
    translation = np.asarray(pose["T"], dtype=float).reshape(3)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
      raise ValueError("invalid camera pose: {}".format(name))
    cameras.append((name, -rotation.T @ translation, rotation.T[:, 2]))
  if not cameras:
    raise ValueError("no calibrated cameras to plot")
  return cameras


def render_camera_layout(source, destination=None, court=None):
  """Render only cameras present in the JSON, never invented reference poses."""
  from multical.app.worldpoints import court_geometry
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure
  from matplotlib.font_manager import FontProperties, findfont
  from matplotlib.patches import Rectangle

  source = Path(source)
  destination = Path(destination) if destination else source.with_suffix(".png")
  data = json.loads(source.read_text(encoding="utf-8"))
  cameras = camera_poses(data)
  court_type = court if court is not None else data.get("court", "tennis")
  geometry = court_geometry(court_type)
  font = None
  for family in ("PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "SimHei"):
    try:
      font = FontProperties(fname=findfont(family, fallback_to_default=False))
      break
    except ValueError:
      pass
  chinese = font is not None
  font = font or FontProperties(family="DejaVu Sans")
  bg, court, ink = "#b9cdbb", "#48756b", "#253d38"
  white, muted = "#ffffff", "#526a60"
  colors = ["#b96b18", "#287bb0", "#bd5679", "#8561b3", "#a7892c"]
  # Match the reference camera identities where present.
  palette = {"cam0": colors[0], "cam1": colors[1], "cam2": "#198b7b",
             "cam3": colors[4], "cam4": colors[2], "cam5": colors[3]}
  fig = Figure(figsize=(18.6, max(7.6, 5.6 + len(cameras) * .4)), dpi=120,
               facecolor=bg)
  FigureCanvasAgg(fig)
  ax = fig.add_axes([.008, .12, .984, .70], facecolor=bg)
  length, doubles = geometry["length"], geometry["doubles"] / 2
  singles, service = geometry["singles"] / 2, geometry["service"]
  ax.add_patch(Rectangle((0, -doubles), length, 2*doubles, facecolor=court, zorder=0))
  if court_type == "tennis":
    ax.add_patch(Rectangle((service, -singles), length-2*service, 2*singles,
                           facecolor="#527e73", zorder=0))
  def line(xs, ys, width=2):
    ax.plot(xs, ys, color=white, lw=width, zorder=1, solid_capstyle="butt")
  for x1, y1, x2, y2 in geometry["lines"]:
    line([x1, x2], [y1, y2])
  if court_type == "tennis":
    for x in (0, length):
      line([x, x + (.45 if x == 0 else -.45)], [0, 0])
  # The net is drawn separately from the painted floor markings.
  line([length/2, length/2], [-geometry["net_half"], geometry["net_half"]])
  positions = np.array([c[1] for c in cameras])
  xmin, xmax = min(0, positions[:, 0].min())-.7, max(length, positions[:, 0].max())+.7
  ymin, ymax = min(-doubles, positions[:, 1].min())-1, max(doubles, positions[:, 1].max())+1
  figure_height = fig.get_figheight()
  fig.set_figwidth(figure_height * .70 * (xmax-xmin) / (ymax-ymin) / .984)
  ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
  ax.axis("off")
  for i, (name, position, forward) in enumerate(cameras):
    x, y, z = position
    color = palette.get(name, colors[i % len(colors)])
    # Orthogonal offsets to the nearest baseline and doubles sideline.
    # Dashed extensions make the reference explicit for cameras beyond corners.
    baseline = 0 if x <= length/2 else length
    sideline = doubles if y >= 0 else -doubles
    ax.plot([x, baseline], [y, y], color=color, lw=1.5,
            linestyle=(0, (4, 3)), zorder=2)
    ax.plot([x, x], [y, sideline], color=color, lw=1.5,
            linestyle=(0, (4, 3)), zorder=2)
    ax.plot([baseline, baseline], [y, sideline], color=muted,
            lw=1, linestyle=(0, (2, 4)), alpha=.4, zorder=2)
    ax.plot([x, baseline], [sideline, sideline], color=muted,
            lw=1, linestyle=(0, (2, 4)), alpha=.4, zorder=2)
    ax.plot([baseline], [y], marker="|", color=color, ms=9, zorder=3)
    ax.plot([x], [sideline], marker="_", color=color, ms=9, zorder=3)
    label_box = dict(boxstyle="round,pad=.25", fc=white, ec="#e0e7e3", lw=.6)
    ax.annotate(f"X: {abs(x-baseline):.2f} m", ((x+baseline)/2, y),
                xytext=(0, 10 if y >= 0 else -18), textcoords="offset points",
                ha="center", color=ink, fontsize=9, bbox=label_box, zorder=5)
    ax.annotate(f"Y: {abs(y-sideline):.2f} m", (x, (y+sideline)/2),
                xytext=(14 if x <= length/2 else -14, -36 if y >= 0 else 36),
                textcoords="offset points", va="center",
                ha="left" if x <= length/2 else "right",
                color=ink, fontsize=9, bbox=label_box, zorder=5)
    direction = forward[:2]
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
      dx, dy = direction / norm * 1.7
      ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                  mutation_scale=18), zorder=3)
    ax.scatter([x], [y], s=340, c=[color], alpha=.10, edgecolors="none", zorder=3)
    ax.scatter([x], [y], s=110, c=[color], edgecolors=white, linewidths=2.5, zorder=4)
    ax.annotate(name, (x, y), xytext=(0, 15 if y >= 0 else -25),
                textcoords="offset points", ha="center", color=ink,
                fontsize=10, fontproperties=font, zorder=5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=.2", fc=bg, ec="none"))
  # Keep coordinate text outside the metric plot, ordered by camera X.
  # Separate rows prevent labels for nearby stereo cameras from colliding.
  for upper in (True, False):
    row = sorted((c for c in cameras if (c[1][1] >= 0) == upper), key=lambda c:c[1][0])
    for index, (name, position, _) in enumerate(row):
      color = palette.get(name, colors[index % len(colors)])
      x, y, z = position
      label_x = .5 if len(row) == 1 else .17 + .66 * index / (len(row)-1)
      fig.text(label_x, .845 if upper else .095,
               f"{name}   X={x:+.2f}   Y={y:+.2f}   Z={z:.2f} m",
               ha="center", va="center", color=color, fontsize=14,
               fontproperties=FontProperties(family="DejaVu Sans"))
  ax.scatter([0], [0], s=45, c=ink, edgecolors=white, zorder=4)
  for end, label in [((2.2, 0), "+X"), ((0, 2.2), "+Y")]:
    ax.annotate("", xy=end, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#d7e9df", lw=1.6))
    ax.text(end[0]+.15, end[1]+.15, label, color="#d7e9df", fontsize=11)
  ax.annotate("O (0, 0)", (0, 0), xytext=(-10, -20), textcoords="offset points",
              ha="right", color=ink, fontsize=9)
  title = (geometry["label"] + " · 相机布局与场地距离" if chinese
           else court_type.capitalize() + " · Camera layout & court offsets")
  subtitle = (f"{len(cameras)} 台相机 · 世界坐标标定结果 · 原点：左底线中心 · +Y 向上" if chinese
              else f"{len(cameras)} calibrated cameras · Origin: left baseline center · +Y up")
  note = ("虚线 X / Y：至最近底线 / 双打边线（含延长线）的距离，仅计 XY；实线箭头：光轴 XY 投影。" if chinese
          else "Dashed X / Y: offsets to nearest baseline / doubles sideline, including extensions; arrows: optical axes in XY.")
  fig.text(.5, .978, title, ha="center", va="top", color=ink, fontsize=21,
           fontproperties=font, fontweight="bold")
  fig.text(.5, .914, source.parent.name + " · " + subtitle,
           ha="center", color=muted, fontsize=12, fontproperties=font)
  fig.text(.5, .048, note, ha="center", color=muted, fontsize=10, fontproperties=font)
  fig.text(.5, .018, f"SOURCE  {source.parent.name}/{source.name}  ·  XY / m",
           ha="center", color=muted, fontsize=9)
  destination.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(destination, facecolor=bg)
  return destination
