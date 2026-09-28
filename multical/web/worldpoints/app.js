let COURT = null;
const $ = (id) => document.getElementById(id);

let state = null;
let activeName = null;
let draft = { x: 0, y: 0, z: 0 };
const DEFAULT_VIEW = { yaw: -0.58, pitch: 0.52, zoom: 1 };
let view = { ...DEFAULT_VIEW };
let dragging = null;

const elements = {
  list: $("pointList"), progress: $("progress"), output: $("outputPath"),
  workListMode: $("workListMode"), workListTitle: $("workListTitle"),
  mode: $("modeBadge"), saveStatus: $("saveStatus"), name: $("currentName"),
  currentState: $("currentState"), x: $("coordX"), y: $("coordY"), z: $("coordZ"),
  validation: $("validationMessage"), markerFields: $("markerFields"),
  markerIds: $("markerIds"), markerHeights: $("markerHeights"),
  markerOccurrences: $("markerOccurrences"), observeImage: $("observeImage"),
  observeEmpty: $("observeEmpty"), observeTitle: $("observeTitle"),
  observeMeta: $("observeMeta"), topCard: $("topCard"),
  exportTop: $("exportTop"), fullscreenTop: $("fullscreenTop"),
  courtCard: $("courtCard"),
  court3d: $("court3d"), fullscreen3d: $("fullscreen3d"),
  previewKicker: $("previewKicker"), top: $("topView"),
  save: $("saveButton"),
  undo: $("undoButton"), clear: $("clearButton"), reset3d: $("reset3d")
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function valuesFor(name) {
  if (state.mode === "measured") return state.points[name] || null;
  const location = state.locations.find((item) => item.name === name);
  if (!location) return null;
  const z = state.marker_specs.length ? state.marker_specs[0].height : 0;
  return [location.x, location.y, z];
}

function isComplete(name) { return valuesFor(name) !== null; }

function coordinateText(name) {
  const values = valuesFor(name);
  if (!values) return null;
  const count = state.mode === "measured" ? 3 : 2;
  return `[${values.slice(0, count).map((value) => (
    Number(value).toFixed(3)
  )).join(", ")}]`;
}

function nextSuggestedName() {
  return state.point_names.find((name) => !isComplete(name)) || state.point_names[0];
}

function selectPoint(name, showObserve = true) {
  if (!name) return;
  activeName = name;
  const values = valuesFor(name) || [0, 0, 0];
  draft = { x: values[0], y: values[1], z: values[2] };
  syncInputs();
  renderAll();
  if (showObserve && state.has_images) {
    const markerImages = state.image_mode === "marker_captures";
    elements.observeTitle.textContent = markerImages ? `${name} · Marker 图片` : `${name} · 手戳结果`;
    elements.observeMeta.textContent = markerImages ? state.cameras.join(" · ") : "只读，不修改 observe";
    elements.observeEmpty.style.display = "none";
    elements.observeImage.style.display = "block";
    elements.observeImage.src = `/api/preview?point=${encodeURIComponent(name)}&t=${Date.now()}`;
  }
}

function syncInputs() {
  elements.name.textContent = activeName || "—";
  elements.x.value = Number(draft.x).toFixed(3);
  elements.y.value = Number(draft.y).toFixed(3);
  elements.z.value = Number(draft.z).toFixed(3);
  const complete = activeName && isComplete(activeName);
  elements.currentState.textContent = complete ? "已设置" : "未设置";
  elements.currentState.classList.toggle("complete", Boolean(complete));
}

function renderPointList() {
  elements.list.replaceChildren();
  const completeCount = state.point_names.filter(isComplete).length;
  elements.progress.textContent = `${completeCount} / ${state.point_names.length}`;
  state.point_names.forEach((name) => {
    const button = document.createElement("button");
    button.className = "point-button";
    if (name === activeName) button.classList.add("active");
    if (isComplete(name)) button.classList.add("complete");
    const label = document.createElement("span");
    label.className = "point-label";
    const pointName = document.createElement("span");
    pointName.className = "point-name";
    pointName.textContent = name;
    label.append(pointName);
    const coordinates = coordinateText(name);
    if (coordinates) {
      const coordinate = document.createElement("span");
      coordinate.className = "point-coordinate";
      coordinate.textContent = coordinates;
      label.append(coordinate);
    }
    const mark = document.createElement("span");
    mark.className = "status-mark";
    button.append(label, mark);
    button.addEventListener("click", () => selectPoint(name));
    elements.list.append(button);
  });
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function allCoordinates() {
  if (!state) return [];
  if (state.mode === "measured") {
    return Object.entries(state.points).map(([name, p]) => ({ name, x: p[0], y: p[1], z: p[2] }));
  }
  const points = [];
  state.locations.forEach((location) => state.marker_specs.forEach((spec) => {
    points.push({ name: `${location.name}:${spec.marker_id}`, x: location.x, y: location.y, z: spec.height });
  }));
  return points;
}

function pointLabels() {
  if (!state) return [];
  if (state.mode === "measured") return allCoordinates();
  const labelHeight = Math.max(
    0, ...state.marker_specs.map((spec) => Number(spec.height) || 0)
  );
  return state.locations.map((location) => ({
    name: location.name, x: location.x, y: location.y, z: labelHeight
  }));
}

function drawPointLabel(
    ctx, x, y, label, active = false, large = false, compact = false,
    position = null) {
  ctx.save();
  ctx.font = compact
    ? "700 9px ui-monospace, SFMono-Regular, Menlo, monospace"
    : large
      ? "800 12px ui-monospace, SFMono-Regular, Menlo, monospace"
      : "800 10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.shadowColor = "rgba(9,28,20,.72)";
  ctx.shadowBlur = 2.5; ctx.shadowOffsetY = 1;
  ctx.fillStyle = active ? "#ff8a4c" : "#fff1a8";
  const offset = compact ? 4 : large ? 7 : 5;
  ctx.fillText(
    label, position ? position.x : x + offset,
    position ? position.y : y - offset
  );
  ctx.restore();
}

function drawExportPointLabels(ctx, points, px, py, width, height) {
  const placed = [];
  ctx.save();
  ctx.font = "700 9px ui-monospace, SFMono-Regular, Menlo, monospace";
  points.forEach((point) => {
    const label = `${point.name} ${coordinateText(point.name)}`;
    const pointX = px(point.x), pointY = py(point.y);
    const labelWidth = ctx.measureText(label).width;
    const labelHeight = 11;
    const candidates = [];
    [-5, 16, -19, 30, -33, 44].forEach((offsetY) => {
      candidates.push({x:pointX+5,y:pointY+offsetY});
      candidates.push({x:pointX-labelWidth-5,y:pointY+offsetY});
    });
    const boxes = candidates.map((candidate) => {
      const x = Math.max(3, Math.min(width-labelWidth-3, candidate.x));
      const y = Math.max(labelHeight+3, Math.min(height-3, candidate.y));
      return {x,y,left:x-3,right:x+labelWidth+3,
        top:y-labelHeight-3,bottom:y+3};
    });
    const overlaps = (box) => placed.some((other) => !(
      box.right < other.left || box.left > other.right ||
      box.bottom < other.top || box.top > other.bottom
    ));
    const index = Math.max(0, boxes.findIndex((box) => !overlaps(box)));
    const position = boxes[index];
    if (index > 0) {
      ctx.beginPath();ctx.moveTo(pointX,pointY);
      ctx.lineTo(position.x,position.y-labelHeight/2);
      ctx.strokeStyle="rgba(255,241,168,.68)";ctx.lineWidth=.7;ctx.stroke();
    }
    drawPointLabel(
      ctx,pointX,pointY,label,false,false,true,position
    );
    placed.push(position);
  });
  ctx.restore();
}

function courtAnchors() {
  const halfDoubles = COURT.doubles / 2, halfSingles = COURT.singles / 2;
  const anchors = [];
  // Endpoints and actual painted-line intersections are snap targets.
  COURT.lines.forEach(([x0,y0,x1,y1]) => {
    anchors.push({x:x0,y:y0,inner:true}, {x:x1,y:y1,inner:true});
    if (y0 !== y1) return;
    COURT.lines.forEach(([vx,vy0,vx1,vy1]) => {
      if (vx === vx1 && vx >= Math.min(x0,x1) && vx <= Math.max(x0,x1)
          && y0 >= Math.min(vy0,vy1) && y0 <= Math.max(vy0,vy1)) {
        anchors.push({x:vx,y:y0,inner:true});
      }
    });
  });
  [0, COURT.net_x, COURT.length].forEach((x) => {
    [-halfDoubles,-halfSingles,0,halfSingles,halfDoubles].forEach((y) => {
      anchors.push({x,y,inner:Math.abs(y) !== halfDoubles || x === COURT.net_x});
    });
  });
  [...(COURT.snap_anchors || []), ...(state?.snap_anchors || [])].forEach((point) => {
    anchors.push({x:Number(point.x), y:Number(point.y), inner:true, configured:true});
  });
  const unique = new Map();
  anchors.forEach((anchor) => {
    unique.set(`${anchor.x.toFixed(9)},${anchor.y.toFixed(9)}`, anchor);
  });
  return Array.from(unique.values());
}

function drawTop(exportCoordinates = false) {
  if (!COURT) return;
  const { ctx, width: w, height: h } = setupCanvas(elements.top);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#174f39"; ctx.fillRect(0, 0, w, h);
  const pad = 22, leftPad = 76, xmin = -0.8, xmax = COURT.length + 0.8;
  const ymin = -COURT.doubles / 2 - 1.55, ymax = COURT.doubles / 2 + 0.8;
  const px = (x) => leftPad + (x - xmin) / (xmax - xmin) * (w - leftPad - pad);
  const py = (y) => h - pad - (y - ymin) / (ymax - ymin) * (h - pad * 2);
  ctx.fillStyle = "#1d6b4a";
  ctx.fillRect(px(0), py(COURT.doubles/2), px(COURT.length)-px(0), py(-COURT.doubles/2)-py(COURT.doubles/2));
  ctx.strokeStyle = "rgba(255,255,255,.98)"; ctx.lineWidth = 1.7;
  const line = (x1, y1, x2, y2) => { ctx.beginPath(); ctx.moveTo(px(x1), py(y1)); ctx.lineTo(px(x2), py(y2)); ctx.stroke(); };
  COURT.lines.forEach((segment) => line(...segment));

  ctx.save();
  ctx.strokeStyle="#b9d4c6"; ctx.fillStyle="#e6f2ec"; ctx.lineWidth=1;
  ctx.font="700 10px ui-monospace, monospace";
  const xAxisY=-COURT.doubles/2-.65;
  line(0,xAxisY,COURT.length,xAxisY);
  ctx.textAlign="center";ctx.textBaseline="top";
  [0, COURT.service, COURT.net_x, COURT.length-COURT.service, COURT.length].map((x) => [x,x.toFixed(3)]).forEach(([x,label])=>{
    ctx.beginPath();ctx.moveTo(px(x),py(xAxisY)-4);ctx.lineTo(px(x),py(xAxisY)+4);ctx.stroke();
    ctx.fillText(label,px(x),py(xAxisY)+7);
  });
  ctx.fillText("X (m)",px(COURT.length/2),py(xAxisY)+25);
  const yAxisPixel=px(0)-18;
  ctx.beginPath();
  ctx.moveTo(yAxisPixel,py(-COURT.doubles/2));
  ctx.lineTo(yAxisPixel,py(COURT.doubles/2));
  ctx.stroke();
  ctx.textAlign="right";ctx.textBaseline="bottom";
  ctx.fillText("Y (m)",yAxisPixel,py(COURT.doubles/2)-10);
  ctx.textAlign="right";ctx.textBaseline="middle";
  [-COURT.doubles/2,-COURT.singles/2,0,COURT.singles/2,COURT.doubles/2].map((y) => [y,y.toFixed(3)]).forEach(([y,label])=>{
    ctx.beginPath();ctx.moveTo(yAxisPixel-4,py(y));ctx.lineTo(yAxisPixel+4,py(y));ctx.stroke();
    ctx.fillText(y > 0 ? `+${label}` : label,yAxisPixel-7,py(y));
  });
  ctx.restore();

  if (!exportCoordinates) {
    courtAnchors().forEach((anchor) => {
      const x=px(anchor.x), y=py(anchor.y);
      ctx.beginPath(); ctx.arc(x,y,anchor.inner?2.5:1.5,0,Math.PI*2);
      ctx.fillStyle=anchor.inner?"#ffd166":"#bde5d1"; ctx.fill();
      ctx.strokeStyle="#173e2e"; ctx.lineWidth=.75; ctx.stroke();
    });
  }
  allCoordinates().forEach((p) => {
    const active = !exportCoordinates && p.name.startsWith(activeName || "\0");
    drawPoint2d(
      ctx, px(p.x), py(p.y), active, exportCoordinates ? .8 : 1.45
    );
  });
  const labels = pointLabels();
  if (exportCoordinates) {
    drawExportPointLabels(ctx,labels,px,py,w,h);
  } else {
    labels.forEach((p) => drawPointLabel(
      ctx,px(p.x),py(p.y),p.name,p.name===activeName,true
    ));
  }
  if (activeName && !exportCoordinates) {
    drawDraft(ctx, px(draft.x), py(draft.y));
  }
  elements.top._map = { px, py, xmin, xmax, ymin, ymax, pad, leftPad, w, h, anchors:courtAnchors() };
}

function drawPoint2d(ctx, x, y, active, scale = 1) {
  ctx.beginPath(); ctx.arc(x, y, (active ? 5 : 3.5) * scale, 0, Math.PI * 2);
  ctx.fillStyle = active ? "#fff" : "#ed6b2d"; ctx.fill();
  if (active) { ctx.strokeStyle = "#ed6b2d"; ctx.lineWidth = 2.5 * scale; ctx.stroke(); }
}

function drawDraft(ctx, x, y) {
  const size = 10;
  ctx.save(); ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x-size, y); ctx.lineTo(x+size, y);
  ctx.moveTo(x, y-size); ctx.lineTo(x, y+size);
  ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = 5; ctx.stroke();
  ctx.strokeStyle = "#06b6d4"; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI*2);
  ctx.fillStyle = "#06b6d4"; ctx.fill();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
  ctx.restore();
}

function draw3d() {
  if (!COURT) return;
  const { ctx, width: w, height: h } = setupCanvas(elements.court3d);
  ctx.clearRect(0, 0, w, h);
  const backdrop = ctx.createLinearGradient(0, 0, 0, h);
  backdrop.addColorStop(0, "#f7faf8");
  backdrop.addColorStop(1, "#dce6e0");
  ctx.fillStyle = backdrop;
  ctx.fillRect(0, 0, w, h);
  const scale = Math.min(w / (COURT.length + 7.23), h / (COURT.doubles + 5.03)) * view.zoom;
  const project = (x, y, z = 0) => {
    // Match the top view: from the origin, positive Y is the upper court side.
    const dx = x - COURT.length / 2, dy = -y;
    const u = dx * Math.cos(view.yaw) - dy * Math.sin(view.yaw);
    const depth = dx * Math.sin(view.yaw) + dy * Math.cos(view.yaw);
    return [
      w/2 + u*scale,
      h*.59 + (depth*Math.sin(view.pitch) - z*Math.cos(view.pitch))*scale,
      depth
    ];
  };
  const path = (points, close = false) => {
    ctx.beginPath();
    points.forEach((point, index) => {
      const screen = project(...point);
      if (index) ctx.lineTo(screen[0], screen[1]);
      else ctx.moveTo(screen[0], screen[1]);
    });
    if (close) ctx.closePath();
  };
  const fillPolygon = (points, color) => {
    path(points, true); ctx.fillStyle = color; ctx.fill();
  };
  const line = (a, b, color="#f7faf8", width=1.3) => {
    const p=project(...a), q=project(...b);
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]);
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke();
  };
  const drawDimensionScale = (marks, side = 1, stagger = false) => {
    const screens=marks.map(([point,label])=>({point:project(...point),label}));
    const p=screens[0].point,q=screens[screens.length-1].point;
    const dx=q[0]-p[0],dy=q[1]-p[1];
    const length=Math.max(1,Math.hypot(dx,dy));
    const nx=-dy/length,ny=dx/length,tick=4;
    ctx.save();ctx.strokeStyle="rgba(31,54,44,.72)";ctx.lineWidth=.9;
    ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);
    screens.forEach(({point})=>{
      ctx.moveTo(point[0]-nx*tick,point[1]-ny*tick);
      ctx.lineTo(point[0]+nx*tick,point[1]+ny*tick);
    });
    ctx.stroke();
    ctx.font="700 9px ui-monospace, monospace";
    ctx.textAlign="center";ctx.textBaseline="middle";
    screens.forEach(({point,label},index)=>{
      const offset=side*(11+(stagger&&index%2?12:0));
      const x=point[0]+nx*offset,y=point[1]+ny*offset;
      const textWidth=ctx.measureText(label).width;
      ctx.fillStyle="rgba(247,250,248,.86)";
      ctx.fillRect(x-textWidth/2-3,y-7,textWidth+6,14);
      ctx.fillStyle="#253f34";ctx.fillText(label,x,y);
    });
    ctx.restore();
  };

  const runoff = [
    [-2.7,(-COURT.doubles/2-1.665),0],[COURT.length+2.7,(-COURT.doubles/2-1.665),0],
    [COURT.length+2.7,(COURT.doubles/2+1.665),0],[-2.7,(COURT.doubles/2+1.665),0]
  ];
  ctx.save(); ctx.translate(0,5); fillPolygon(runoff,"rgba(20,42,33,.18)"); ctx.restore();
  fillPolygon(runoff,"#b86142");
  path(runoff,true); ctx.strokeStyle="#8e452f"; ctx.lineWidth=1.2; ctx.stroke();

  const court = [
    [0,-COURT.doubles/2,0],[COURT.length,-COURT.doubles/2,0],
    [COURT.length,COURT.doubles/2,0],[0,COURT.doubles/2,0]
  ];
  fillPolygon(court,"#19704c");
  fillPolygon([
    [COURT.service,-COURT.singles/2,0],[COURT.net_x,-COURT.singles/2,0],
    [COURT.net_x,0,0],[COURT.service,0,0]
  ],"rgba(255,255,255,.045)");
  fillPolygon([
    [COURT.net_x,0,0],[COURT.length-COURT.service,0,0],
    [COURT.length-COURT.service,COURT.singles/2,0],[COURT.net_x,COURT.singles/2,0]
  ],"rgba(255,255,255,.045)");

  COURT.lines.forEach(([x0,y0,x1,y1]) => line([x0,y0,0],[x1,y1,0]));

  const netX=COURT.net_x, netHalf=COURT.net_half;
  const netTop=(y)=>COURT.net_center+Math.abs(y)/netHalf*(COURT.net_post-COURT.net_center);
  const netFace=[];
  for(let i=0;i<=16;i++) { const y=-netHalf+i*netHalf/8; netFace.push([netX,y,netTop(y)]); }
  for(let i=16;i>=0;i--) netFace.push([netX,-netHalf+i*netHalf/8,COURT.net_bottom]);
  fillPolygon(netFace,"rgba(238,245,241,.56)");
  for(let i=0;i<=16;i++) {
    const y=-netHalf+i*netHalf/8;
    line([netX,y,COURT.net_bottom],[netX,y,netTop(y)],"rgba(61,79,70,.32)",.65);
  }
  [.25,.5,.75].map((t)=>COURT.net_bottom+t*(COURT.net_center-COURT.net_bottom)).forEach((z)=>line(
    [netX,-netHalf,z],[netX,netHalf,z],"rgba(61,79,70,.32)",.65
  ));
  path(Array.from({length:17},(_,i)=>{
    const y=-netHalf+i*(netHalf*2/16); return [netX,y,netTop(y)];
  }));
  ctx.strokeStyle="#f8fbf9";ctx.lineWidth=2;ctx.stroke();
  line([netX,-netHalf,0],[netX,-netHalf,COURT.net_post],"#243b32",3);
  line([netX,netHalf,0],[netX,netHalf,COURT.net_post],"#243b32",3);

  const dimensionColor="rgba(31,54,44,.42)";
  line([0,-COURT.doubles/2,0],[0,(-COURT.doubles/2-.765),0],dimensionColor,.7);
  line([COURT.length,-COURT.doubles/2,0],[COURT.length,(-COURT.doubles/2-.765),0],dimensionColor,.7);
  drawDimensionScale([
    [[0,(-COURT.doubles/2-.765),.02],"0"],
    [[COURT.net_x,(-COURT.doubles/2-.765),.02],COURT.net_x.toFixed(3)],
    [[COURT.length,(-COURT.doubles/2-.765),.02],COURT.length.toFixed(3)]
  ]);
  line([0,-COURT.doubles/2,0],[-1.45,-COURT.doubles/2,0],dimensionColor,.7);
  line([0,COURT.doubles/2,0],[-1.45,COURT.doubles/2,0],dimensionColor,.7);
  drawDimensionScale([
    [[-1.45,-COURT.doubles/2,.02],(-COURT.doubles/2).toFixed(3)],
    [[-1.45,0,.02],"0"],
    [[-1.45,COURT.doubles/2,.02],(COURT.doubles/2).toFixed(3)]
  ],-1);

  allCoordinates().slice().sort((a,b)=>project(a.x,a.y,a.z)[2]-project(b.x,b.y,b.z)[2]).forEach((p) => {
    const ground=project(p.x,p.y,0), q=project(p.x,p.y,p.z);
    ctx.save();ctx.setLineDash([3,3]);line([p.x,p.y,0],[p.x,p.y,p.z],"rgba(18,42,32,.38)",1);ctx.restore();
    ctx.beginPath();ctx.ellipse(ground[0],ground[1],4,2,0,0,Math.PI*2);ctx.fillStyle="rgba(15,35,27,.24)";ctx.fill();
    drawPoint2d(ctx,q[0],q[1],p.name.startsWith(activeName||"\0"));
  });
  pointLabels().forEach((p) => {
    const q=project(p.x,p.y,p.z);
    drawPointLabel(ctx,q[0],q[1],p.name,p.name===activeName);
  });
  if (activeName) { const p=project(draft.x,draft.y,draft.z); drawDraft(ctx,p[0],p[1]); }
}

function renderAll() {
  renderPointList(); draw3d(); drawTop();
  elements.undo.disabled = !state.can_undo;
}

function canvasPosition(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

elements.top.addEventListener("click", (event) => {
  if (!activeName) return;
  const pos = canvasPosition(event, elements.top), m = elements.top._map;
  draft.x = m.xmin + (pos.x-m.leftPad)/(m.w-m.leftPad-m.pad)*(m.xmax-m.xmin);
  draft.y = m.ymax - (pos.y-m.pad)/(m.h-m.pad*2)*(m.ymax-m.ymin);
  const nearest = m.anchors.map((anchor) => ({
    anchor, distance:Math.hypot(pos.x-m.px(anchor.x),pos.y-m.py(anchor.y))
  })).sort((a,b)=>a.distance-b.distance)[0];
  if (nearest && nearest.distance <= 16) {
    draft.x=nearest.anchor.x; draft.y=nearest.anchor.y;
    elements.saveStatus.textContent=`已吸附角点 [${draft.x.toFixed(3)}, ${draft.y.toFixed(3)}]`;
  }
  draft.x = Math.round(draft.x*1000)/1000; draft.y = Math.round(draft.y*1000)/1000;
  syncInputs(); draw3d(); drawTop();
});

elements.court3d.addEventListener("pointerdown", (event) => {
  dragging = { x:event.clientX, y:event.clientY, yaw:view.yaw, pitch:view.pitch };
  elements.court3d.setPointerCapture(event.pointerId);
});
elements.court3d.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  view.yaw = dragging.yaw + (event.clientX-dragging.x)*.008;
  view.pitch = Math.max(.15, Math.min(1.15, dragging.pitch + (event.clientY-dragging.y)*.006));
  draw3d();
});
elements.court3d.addEventListener("pointerup", () => { dragging = null; });
elements.court3d.addEventListener("wheel", (event) => {
  const zoomRequested = event.altKey || document.fullscreenElement === elements.courtCard;
  if (!zoomRequested) return;
  event.preventDefault(); view.zoom = Math.max(.65, Math.min(1.8, view.zoom * (event.deltaY>0?.92:1.08))); draw3d();
}, { passive:false });
elements.reset3d.addEventListener("click", () => {
  view = { ...DEFAULT_VIEW }; draw3d();
});
async function toggleFullscreen(panel) {
  try {
    if (document.fullscreenElement === panel) {
      await document.exitFullscreen();
    } else {
      await panel.requestFullscreen();
    }
  } catch (error) {
    elements.validation.textContent = `无法切换全屏：${error.message}`;
  }
}
elements.exportTop.addEventListener("click", () => {
  drawTop(true);
  elements.top.toBlob((blob) => {
    drawTop();
    if (!blob) {
      elements.validation.textContent = "Top View 图片导出失败";
      return;
    }
    const source = String(state.output || "worldpoints");
    const basename = source.split(/[\\/]/).pop().replace(/\.ya?ml$/i, "");
    const filename = `${basename}_topview.png`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename; link.click();
    URL.revokeObjectURL(url);
    elements.saveStatus.textContent = `已导出 ${filename}`;
  }, "image/png");
});
elements.fullscreenTop.addEventListener("click", () => {
  toggleFullscreen(elements.topCard);
});
elements.fullscreen3d.addEventListener("click", () => {
  toggleFullscreen(elements.courtCard);
});
document.addEventListener("fullscreenchange", () => {
  const topActive = document.fullscreenElement === elements.topCard;
  const courtActive = document.fullscreenElement === elements.courtCard;
  elements.fullscreenTop.textContent = topActive ? "退出全屏" : "全屏";
  elements.fullscreenTop.setAttribute("aria-pressed", String(topActive));
  elements.fullscreen3d.textContent = courtActive ? "退出全屏" : "全屏";
  elements.fullscreen3d.setAttribute("aria-pressed", String(courtActive));
  requestAnimationFrame(() => { drawTop(); draw3d(); });
});

[elements.x, elements.y, elements.z].forEach((input, index) => input.addEventListener("input", () => {
  const value = Number(input.value); if (Number.isFinite(value)) draft[["x","y","z"][index]] = value;
  draw3d(); drawTop();
}));

elements.save.addEventListener("click", async () => {
  if (!activeName) return;
  elements.validation.textContent = ""; elements.save.disabled = true;
  try {
    const currentIndex = state.point_names.indexOf(activeName);
    const payload = { name:activeName, x:draft.x, y:draft.y, z:draft.z };
    if (state.mode === "markers") Object.assign(payload, {
      marker_ids: elements.markerIds.value,
      marker_heights: elements.markerHeights.value,
      marker_occurrences: elements.markerOccurrences.value
    });
    state = await request("/api/save", { method:"POST", body:JSON.stringify(payload) });
    elements.saveStatus.textContent = `已保存 ${activeName}`;
    const next = state.point_names[currentIndex+1] || nextSuggestedName() || activeName;
    selectPoint(next);
  } catch (error) { elements.validation.textContent = error.message; }
  finally { elements.save.disabled = false; }
});

elements.undo.addEventListener("click", async () => {
  try { state = await request("/api/undo", {method:"POST", body:"{}"}); selectPoint(activeName || nextSuggestedName(), false); elements.saveStatus.textContent="已撤销"; }
  catch (error) { elements.validation.textContent=error.message; }
});
elements.clear.addEventListener("click", async () => {
  if (!confirm("确定清空所有已设置的世界坐标吗？observe 不会被修改。")) return;
  try { state=await request("/api/clear",{method:"POST",body:"{}"}); selectPoint(state.point_names[0], false); elements.saveStatus.textContent="已清空"; }
  catch (error) { elements.validation.textContent=error.message; }
});

elements.observeImage.addEventListener("error", () => {
  elements.observeImage.style.display="none"; elements.observeEmpty.style.display="grid";
  elements.observeEmpty.textContent="找不到该点的源图片，请检查 observe 中的 image_path。";
});

window.addEventListener("resize", () => { draw3d(); drawTop(); });

async function initialize() {
  try {
    state = await request("/api/state");
    COURT = state.court;
    $("courtTitle").textContent = `3D ${COURT.label}`;
    if (!document.fullscreenEnabled) {
      elements.fullscreenTop.hidden = true;
      elements.fullscreen3d.hidden = true;
    }
    elements.mode.textContent = state.mode.toUpperCase(); elements.output.textContent = state.output;
    if (state.observe_mode === "single_image_multiple_points") {
      elements.workListMode.textContent = "单图多点 · POINTS";
      elements.workListTitle.textContent = "P01, P02…";
      elements.observeEmpty.textContent = "选择右侧 P 点，查看同一组图片上的对应手戳位置。";
    } else if (state.observe_mode === "multiple_images_single_point") {
      elements.workListMode.textContent = "多图单点 · FRAMES";
      elements.workListTitle.textContent = "图片帧";
      elements.observeEmpty.textContent = "选择右侧图片帧，查看该帧在各相机中的手戳位置。";
    } else if (state.observe_mode === "mixed") {
      elements.workListMode.textContent = "混合 OBSERVE";
      elements.workListTitle.textContent = "标注条目";
    } else if (state.image_mode === "marker_captures") {
      elements.workListMode.textContent = "WORLD MARKERS · CAPTURES";
      elements.workListTitle.textContent = "图片序列";
      elements.observeEmpty.textContent = "选择右侧 capture，查看对应相机的 marker 图片。";
    } else if (state.image_mode === "marker_yaml") {
      elements.workListMode.textContent = "WORLD MARKERS · YAML ONLY";
      elements.workListTitle.textContent = "配置中的 Captures";
      elements.previewKicker.textContent = "MARKER YAML · NO IMAGES";
      elements.observeTitle.textContent = "未加载 Marker 图片";
      elements.observeMeta.textContent = state.cameras.join(" · ");
      elements.observeEmpty.textContent = "已从现有 YAML 加载坐标，可直接选择、修改并保存。";
    } else if (state.image_mode === "measured_yaml") {
      elements.workListMode.textContent = "MEASURED · YAML ONLY";
      elements.workListTitle.textContent = "配置中的 Points";
      elements.previewKicker.textContent = "MEASURED YAML · NO OBSERVE";
      elements.observeTitle.textContent = "未加载 Observe";
      elements.observeMeta.textContent = "自动降级模式";
      elements.observeEmpty.textContent = "已从现有 measured_world_points.yaml 加载坐标，可直接选择、修改并保存。";
    }
    elements.markerFields.hidden = state.mode !== "markers";
    elements.z.closest("label").hidden = state.mode === "markers";
    if (state.mode === "markers" && state.image_mode !== "marker_yaml") {
      elements.previewKicker.textContent = "MARKER IMAGES · READ ONLY";
    }
    elements.markerIds.value = state.marker_specs.map((item)=>item.marker_id).join(",");
    elements.markerHeights.value = state.marker_specs.map((item)=>Number(item.height).toFixed(3)).join(",");
    elements.markerOccurrences.value = state.marker_specs.map((item)=>item.occurrence||"-").join(",");
    elements.undo.disabled = !state.can_undo;
    selectPoint(nextSuggestedName(), false);
  } catch (error) { elements.validation.textContent = error.message; }
}

initialize();
