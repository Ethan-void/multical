import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function setAllBorders(range, color = "#D1D5DB") {
  range.format.borders = {
    top: { style: "thin", color },
    bottom: { style: "thin", color },
    left: { style: "thin", color },
    right: { style: "thin", color },
    insideHorizontal: { style: "thin", color },
    insideVertical: { style: "thin", color },
  };
}

function styleTitle(range) {
  range.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function styleSection(range) {
  range.format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D" },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 22;
}

function styleHeader(range) {
  range.format = {
    fill: "#4472C4",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 30;
  setAllBorders(range, "#AAB7C4");
}

function formulaRange(startRow, endRow, column) {
  return `'逐点误差'!$${column}$${startRow}:$${column}$${endRow}`;
}

function writeSummarySheet(workbook, report) {
  const sheet = workbook.worksheets.getItem("验收汇总");
  const summary = report.summary;
  const acceptance = report.acceptance;
  const units = report.world_units;
  const points = report.points || [];
  const lastPointRow = Math.max(2, points.length + 1);

  sheet.mergeCells("A1:F1");
  sheet.getRange("A1").values = [["三维重建现场精度验收报告"]];
  styleTitle(sheet.getRange("A1:F1"));

  sheet.mergeCells("A3:B3");
  sheet.getRange("A3").values = [["验收结论"]];
  styleSection(sheet.getRange("A3:B3"));
  sheet.getRange("A4:B10").values = [
    ["状态", acceptance.passed ? "通过" : "不通过"],
    ["坐标系", report.coordinate_frame],
    ["世界坐标单位", units],
    ["实测点数", summary.ground_truth_count],
    ["成功匹配", summary.matched_count],
    ["重建失败", summary.failed_reconstruction_count],
    ["缺少重建", summary.missing_reconstruction_count],
  ];
  sheet.getRange("A4").format.font = { bold: true };
  sheet.getRange("B4").format = {
    fill: acceptance.passed ? "#E2F0D9" : "#FCE4D6",
    font: {
      bold: true,
      color: acceptance.passed ? "#006100" : "#9C0006",
    },
    horizontalAlignment: "center",
  };
  setAllBorders(sheet.getRange("A4:B10"));

  sheet.mergeCells("A12:F12");
  sheet.getRange("A12").values = [["三维距离误差"]];
  styleSection(sheet.getRange("A12:F12"));
  sheet.getRange("A13:F13").values = [[
    "指标", "均值", "RMS", "中位数", "P95", "最大值",
  ]];
  styleHeader(sheet.getRange("A13:F13"));
  sheet.getRange("A14").values = [["误差"]];
  if (points.length) {
    const distanceRange = formulaRange(2, lastPointRow, "N");
    sheet.getRange("B14:F14").formulas = [[
      `=AVERAGE(${distanceRange})`,
      `=SQRT(SUMSQ(${distanceRange})/COUNT(${distanceRange}))`,
      `=MEDIAN(${distanceRange})`,
      `=PERCENTILE.INC(${distanceRange},0.95)`,
      `=MAX(${distanceRange})`,
    ]];
  }
  sheet.getRange("B14:F14").format.numberFormat = "0.000000";
  setAllBorders(sheet.getRange("A14:F14"));

  sheet.mergeCells("A16:E16");
  sheet.getRange("A16").values = [["分轴误差"]];
  styleSection(sheet.getRange("A16:E16"));
  sheet.getRange("A17:E17").values = [[
    "轴", "偏差", "MAE", "RMS", "最大绝对误差",
  ]];
  styleHeader(sheet.getRange("A17:E17"));
  const axes = [["X", "H", "K"], ["Y", "I", "L"], ["Z", "J", "M"]];
  axes.forEach(([axis, signedColumn, absoluteColumn], index) => {
    const row = 18 + index;
    sheet.getRange(`A${row}`).values = [[axis]];
    if (points.length) {
      const signedRange = formulaRange(2, lastPointRow, signedColumn);
      const absoluteRange = formulaRange(2, lastPointRow, absoluteColumn);
      sheet.getRange(`B${row}:E${row}`).formulas = [[
        `=AVERAGE(${signedRange})`,
        `=AVERAGE(${absoluteRange})`,
        `=SQRT(SUMSQ(${signedRange})/COUNT(${signedRange}))`,
        `=MAX(${absoluteRange})`,
      ]];
    }
  });
  sheet.getRange("B18:E20").format.numberFormat = "0.000000";
  setAllBorders(sheet.getRange("A18:E20"));

  sheet.mergeCells("A22:D22");
  sheet.getRange("A22").values = [["验收阈值"]];
  styleSection(sheet.getRange("A22:D22"));
  sheet.getRange("A23:D23").values = [[
    "参数", `阈值（${units}）`, `实际值（${units}）`, "结果",
  ]];
  styleHeader(sheet.getRange("A23:D23"));
  const thresholdRows = [
    ["max_mean_error", acceptance.thresholds.max_mean_error, "=B14"],
    ["max_p95_error", acceptance.thresholds.max_p95_error, "=E14"],
    ["max_error", acceptance.thresholds.max_error, "=F14"],
  ];
  thresholdRows.forEach(([name, threshold, actualFormula], index) => {
    const row = 24 + index;
    sheet.getRange(`A${row}:B${row}`).values = [[
      name, threshold === undefined ? null : threshold,
    ]];
    sheet.getRange(`C${row}`).formulas = [[actualFormula]];
    sheet.getRange(`D${row}`).formulas = [[
      threshold === undefined
        ? '="未设置"'
        : `=IF(C${row}<=B${row},"通过","不通过")`,
    ]];
  });
  sheet.getRange("B24:C26").format.numberFormat = "0.000000";
  setAllBorders(sheet.getRange("A24:D26"));

  sheet.mergeCells("A28:F28");
  sheet.getRange("A28").values = [["未通过原因"]];
  styleSection(sheet.getRange("A28:F28"));
  const failures = acceptance.failures || [];
  const failureRows = failures.length ? failures : ["无"];
  failureRows.forEach((failure, index) => {
    const row = 29 + index;
    sheet.mergeCells(`A${row}:F${row}`);
    sheet.getRange(`A${row}`).values = [[failure]];
  });

  const sourceRow = 31 + failureRows.length;
  sheet.mergeCells(`A${sourceRow}:F${sourceRow}`);
  sheet.getRange(`A${sourceRow}`).values = [["输入文件"]];
  styleSection(sheet.getRange(`A${sourceRow}:F${sourceRow}`));
  sheet.getRange(`A${sourceRow + 1}`).values = [["重建结果"]];
  sheet.mergeCells(`B${sourceRow + 1}:F${sourceRow + 1}`);
  sheet.getRange(`B${sourceRow + 1}`).values = [[report.sources.reconstruction]];
  sheet.getRange(`A${sourceRow + 2}`).values = [["实测坐标"]];
  sheet.mergeCells(`B${sourceRow + 2}:F${sourceRow + 2}`);
  sheet.getRange(`B${sourceRow + 2}`).values = [[report.sources.ground_truth]];
  sheet.getRange(`B${sourceRow + 1}:F${sourceRow + 2}`).format.wrapText = true;

  sheet.getRange("A1:F50").format.font = { name: "Aptos", size: 10 };
  styleTitle(sheet.getRange("A1:F1"));
  sheet.getRange("A:A").format.columnWidth = 23;
  sheet.getRange("B:F").format.columnWidth = 17;
  sheet.freezePanes.freezeRows(1);
  return sheet;
}

function writePointSheet(workbook, report) {
  const sheet = workbook.worksheets.getItem("逐点误差");
  const units = report.world_units;
  sheet.getRange("A1:P1").values = [[
    "帧/点编号",
    `实测 X（${units}）`, `实测 Y（${units}）`, `实测 Z（${units}）`,
    `重建 X（${units}）`, `重建 Y（${units}）`, `重建 Z（${units}）`,
    `误差 X（${units}）`, `误差 Y（${units}）`, `误差 Z（${units}）`,
    `|误差 X|（${units}）`, `|误差 Y|（${units}）`, `|误差 Z|（${units}）`,
    `三维误差（${units}）`, "参与相机", "重投影 RMS（px）",
  ]];
  styleHeader(sheet.getRange("A1:P1"));

  (report.points || []).forEach((point, index) => {
    const row = index + 2;
    sheet.getRange(`A${row}:G${row}`).values = [[
      point.frame, ...point.measured_world, ...point.reconstructed_world,
    ]];
    sheet.getRange(`H${row}:N${row}`).formulas = [[
      `=E${row}-B${row}`, `=F${row}-C${row}`, `=G${row}-D${row}`,
      `=ABS(H${row})`, `=ABS(I${row})`, `=ABS(J${row})`,
      `=SQRT(H${row}^2+I${row}^2+J${row}^2)`,
    ]];
    sheet.getRange(`O${row}:P${row}`).values = [[
      Array.isArray(point.cameras_used)
        ? point.cameras_used.join(", ")
        : point.cameras_used ?? "",
      point.reprojection_rms_px ?? null,
    ]];
  });
  const lastRow = Math.max(2, (report.points || []).length + 1);
  sheet.getRange(`B2:N${lastRow}`).format.numberFormat = "0.000000";
  sheet.getRange(`P2:P${lastRow}`).format.numberFormat = "0.000";
  setAllBorders(sheet.getRange(`A1:P${lastRow}`));
  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange("B:N").format.columnWidth = 15;
  sheet.getRange("O:O").format.columnWidth = 26;
  sheet.getRange("P:P").format.columnWidth = 18;
  sheet.getRange(`A1:P${lastRow}`).format.font = {
    name: "Aptos", size: 10,
  };
  styleHeader(sheet.getRange("A1:P1"));
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}

function writeIssueSheet(workbook, report) {
  const sheet = workbook.worksheets.getItem("异常与缺失");
  sheet.getRange("A1:C1").values = [["类型", "帧/点编号", "原因"]];
  styleHeader(sheet.getRange("A1:C1"));
  const rows = [];
  (report.failed_reconstructions || []).forEach((item) => {
    rows.push(["重建失败", item.frame, item.reason]);
  });
  (report.missing_reconstructions || []).forEach((frame) => {
    rows.push(["缺少重建", frame, "实测点没有对应重建结果"]);
  });
  (report.unmeasured_reconstructions || []).forEach((frame) => {
    rows.push(["未实测", frame, "重建结果没有对应实测坐标"]);
  });
  if (!rows.length) rows.push(["无", "", ""]);
  sheet.getRange(`A2:C${rows.length + 1}`).values = rows;
  setAllBorders(sheet.getRange(`A1:C${rows.length + 1}`));
  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange("B:B").format.columnWidth = 24;
  sheet.getRange("C:C").format.columnWidth = 58;
  sheet.getRange(`A1:C${rows.length + 1}`).format.font = {
    name: "Aptos", size: 10,
  };
  styleHeader(sheet.getRange("A1:C1"));
  sheet.freezePanes.freezeRows(1);
  return sheet;
}

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function cameraNames(report) {
  const names = new Set();
  Object.keys(report.camera_groups || {}).forEach((name) => names.add(name));
  (report.points || []).forEach((point) => {
    (point.cameras_used || []).forEach((name) => names.add(name));
    (point.cameras_rejected || []).forEach((name) => names.add(name));
    Object.keys(point.reprojection_errors_px || {}).forEach(
      (name) => names.add(name)
    );
  });
  return [...names].sort((first, second) => first.localeCompare(
    second, undefined, { numeric: true, sensitivity: "base" }
  ));
}

function writeCameraParticipationSheet(workbook, report) {
  const sheet = workbook.worksheets.getItem("相机参与明细");
  const cameras = cameraNames(report);
  const points = report.points || [];
  const cameraGroups = report.camera_groups || {};
  const headers = [
    "点", "观测相机数", "参与相机数", "剔除相机数", "剔除率",
    "观测相机组", "参与相机组", "具备跨组条件", "跨组参与",
    "参与相机", ...cameras,
  ];
  const rows = points.map((point) => {
    const used = point.cameras_used || [];
    const rejected = point.cameras_rejected || [];
    const observedCameras = [...new Set([
      ...used, ...rejected,
    ])];
    const observed = point.camera_observation_count ?? observedCameras.length;
    const errors = point.reprojection_errors_px || {};
    const observedGroups = [...new Set(observedCameras
      .map((camera) => cameraGroups[camera])
      .filter(Boolean))].sort();
    const usedGroups = [...new Set(used
      .map((camera) => cameraGroups[camera])
      .filter(Boolean))].sort();
    const crossGroupEligible = observedGroups.length >= 2;
    const crossGroupParticipating = usedGroups.length >= 2;
    return [
      point.frame,
      observed,
      used.length,
      rejected.length,
      observed ? rejected.length / observed : null,
      observedGroups.length ? observedGroups.join(", ") : "—",
      usedGroups.length ? usedGroups.join(", ") : "—",
      crossGroupEligible ? "是" : "否",
      crossGroupParticipating ? "是" : "否",
      used.join(", "),
      ...cameras.map((camera) => errors[camera] ?? "—"),
    ];
  });
  const tableHeaderRow = 6;
  const firstDataRow = tableHeaderRow + 1;
  const lastDataRow = Math.max(firstDataRow, tableHeaderRow + rows.length);
  const columnCount = headers.length;
  const lastColumn = columnName(columnCount - 1);

  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [["跨组参与汇总"]];
  styleSection(sheet.getRange(`A1:${lastColumn}1`));
  sheet.getRange("A2:C2").values = [[
    "具备跨组观测条件点数", "实际跨组参与点数", "跨组参与率",
  ]];
  styleHeader(sheet.getRange("A2:C2"));
  const eligibleCount = rows.filter((row) => row[7] === "是").length;
  const participatingCount = rows.filter(
    (row) => row[7] === "是" && row[8] === "是"
  ).length;
  sheet.getRange("A3:C3").values = [[
    eligibleCount,
    participatingCount,
    eligibleCount ? participatingCount / eligibleCount : null,
  ]];
  sheet.getRange("A3:B3").format.numberFormat = "0";
  sheet.getRange("C3").format.numberFormat = "0.0%";
  sheet.mergeCells(`A4:${lastColumn}4`);
  sheet.getRange("A4").values = [[
    "定义：跨组参与率 = 实际跨组参与点数 ÷ 具备跨组观测条件点数",
  ]];
  sheet.getRange(`A4:${lastColumn}4`).format = {
    font: { italic: true, color: "#666666" },
    verticalAlignment: "center",
  };

  sheet.getRangeByIndexes(
    tableHeaderRow - 1, 0, 1, columnCount
  ).values = [headers];
  if (rows.length) {
    sheet.getRangeByIndexes(
      firstDataRow - 1, 0, rows.length, columnCount
    ).values = rows;
  }
  styleHeader(sheet.getRange(`A${tableHeaderRow}:${lastColumn}${tableHeaderRow}`));
  setAllBorders(sheet.getRange(`A2:C3`));
  setAllBorders(sheet.getRange(
    `A${tableHeaderRow}:${lastColumn}${lastDataRow}`
  ));
  if (rows.length) {
    sheet.getRange(`B${firstDataRow}:D${lastDataRow}`).format.numberFormat = "0";
    sheet.getRange(`E${firstDataRow}:E${lastDataRow}`).format.numberFormat = "0.0%";
  }
  if (cameras.length && rows.length) {
    const firstCameraColumnIndex = 10;
    const firstCameraColumn = columnName(firstCameraColumnIndex);
    sheet.getRange(
      `${firstCameraColumn}${firstDataRow}:${lastColumn}${lastDataRow}`
    ).format = {
      horizontalAlignment: "center",
      numberFormat: "0.0000",
    };
    points.forEach((point, pointIndex) => {
      const rejected = new Set(point.cameras_rejected || []);
      cameras.forEach((camera, cameraIndex) => {
        if (!rejected.has(camera)) return;
        const cell = sheet.getCell(
          pointIndex + firstDataRow - 1,
          cameraIndex + firstCameraColumnIndex
        );
        const error = (point.reprojection_errors_px || {})[camera];
        if (error === undefined || error === null) {
          cell.values = [["剔除"]];
        } else {
          cell.format.numberFormat = '0.00"×"';
        }
        cell.format.font = { color: "#C00000" };
      });
    });
  }
  sheet.getRange(`A1:${lastColumn}${lastDataRow}`).format.font = {
    name: "Aptos", size: 10,
  };
  styleSection(sheet.getRange(`A1:${lastColumn}1`));
  styleHeader(sheet.getRange("A2:C2"));
  styleHeader(sheet.getRange(`A${tableHeaderRow}:${lastColumn}${tableHeaderRow}`));
  sheet.getRange("A:A").format.columnWidth = 12;
  sheet.getRange("B:E").format.columnWidth = 11;
  sheet.getRange("F:G").format.columnWidth = 24;
  sheet.getRange("H:I").format.columnWidth = 16;
  sheet.getRange("J:J").format.columnWidth = 32;
  if (cameras.length) {
    sheet.getRange(`${columnName(10)}:${lastColumn}`).format.columnWidth = 13;
  }
  sheet.freezePanes.freezeRows(tableHeaderRow);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}

function yesNo(value) {
  if (value === true) return "是";
  if (value === false) return "否";
  return "未提供";
}

function writeBundleAdjustmentSheet(workbook, report) {
  const sheet = workbook.worksheets.getItem("联合BA优化收敛情况");
  const bundle = report.bundle_adjustment || {};
  const joint = bundle.joint || {};
  const groups = bundle.groups || [];

  sheet.mergeCells("A1:I1");
  sheet.getRange("A1").values = [["联合 BA 优化收敛情况"]];
  styleTitle(sheet.getRange("A1:I1"));

  sheet.mergeCells("A3:I3");
  sheet.getRange("A3").values = [["总体状态"]];
  styleSection(sheet.getRange("A3:I3"));
  sheet.getRange("A4:B10").values = [
    ["联合优化收敛", yesNo(joint.optimization_success)],
    ["预优化成功", yesNo(joint.warmup_success)],
    ["优化终止原因", joint.optimization_message || "未提供"],
    ["优化方法", bundle.method || "未提供"],
    ["损失函数", joint.loss || "未提供"],
    ["硬剔除复检", yesNo(joint.hard_outlier_rejection_applied)],
    ["重投影门限（px）", joint.ransac_threshold_px ?? "未提供"],
  ];
  setAllBorders(sheet.getRange("A4:B10"));
  sheet.getRange("A4:A10").format.font = { bold: true };
  sheet.getRange("B4").format = {
    fill: joint.optimization_success === true ? "#E2F0D9" : "#FCE4D6",
    font: {
      bold: true,
      color: joint.optimization_success === true ? "#006100" : "#9C0006",
    },
    horizontalAlignment: "center",
  };
  if (typeof joint.ransac_threshold_px === "number") {
    sheet.getRange("B10").format.numberFormat = "0.000";
  }

  sheet.mergeCells("A12:I12");
  sheet.getRange("A12").values = [["优化前后重投影误差"]];
  styleSection(sheet.getRange("A12:I12"));
  sheet.getRange("A13:H13").values = [[
    "数据类型", "阶段", "数量", "均值（px）", "RMS（px）",
    "中位数（px）", "P95（px）", "最大值（px）",
  ]];
  styleHeader(sheet.getRange("A13:H13"));
  const metricRows = [
    ["Charuco", "优化前", joint.initial_charuco],
    ["Charuco", "优化后", joint.final_charuco],
    ["世界控制点", "优化前（全部）", joint.initial_world],
    ["世界控制点", "优化后（全部）", joint.final_world_all],
    ["世界控制点", "优化后（内点）", joint.final_world_inlier],
  ].map(([type, stage, values]) => [
    type,
    stage,
    values?.count ?? "—",
    values?.mean ?? "—",
    values?.RMS ?? "—",
    values?.median ?? "—",
    values?.p95 ?? "—",
    values?.max ?? "—",
  ]);
  sheet.getRange("A14:H18").values = metricRows;
  setAllBorders(sheet.getRange("A14:H18"));
  sheet.getRange("C14:C18").format.numberFormat = "#,##0";
  sheet.getRange("D14:H18").format.numberFormat = "0.000000";

  sheet.mergeCells("A20:I20");
  sheet.getRange("A20").values = [["分组优化结果"]];
  styleSection(sheet.getRange("A20:I20"));
  sheet.getRange("A21:I21").values = [[
    "相机组", "相机", "优化成功", "观测数", "内点数", "内点率",
    "内点 RMS（px）", "全部点 RMS（px）", "最大误差（px）",
  ]];
  styleHeader(sheet.getRange("A21:I21"));
  if (groups.length) {
    const groupRows = groups.map((group) => [
      group.name || "—",
      (group.cameras || []).join(", ") || "—",
      yesNo(group.optimization_success),
      group.observation_count ?? "—",
      group.inlier_count ?? "—",
      null,
      group.reprojection_rms_px ?? "—",
      group.all_points_rms_px ?? "—",
      group.all_points_max_px ?? "—",
    ]);
    const lastRow = 21 + groupRows.length;
    sheet.getRange(`A22:I${lastRow}`).values = groupRows;
    sheet.getRange(`F22:F${lastRow}`).formulas = groups.map(
      (_, index) => {
        const row = 22 + index;
        return [`=IF(D${row}=0,"",E${row}/D${row})`];
      }
    );
    setAllBorders(sheet.getRange(`A22:I${lastRow}`));
    sheet.getRange(`D22:E${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`F22:F${lastRow}`).format.numberFormat = "0.0%";
    sheet.getRange(`G22:I${lastRow}`).format.numberFormat = "0.000000";
  } else {
    sheet.mergeCells("A22:I22");
    sheet.getRange("A22").values = [["未提供分组优化数据"]];
  }

  sheet.getRange("A1:I24").format.font = { name: "Aptos", size: 10 };
  styleTitle(sheet.getRange("A1:I1"));
  styleSection(sheet.getRange("A3:I3"));
  styleSection(sheet.getRange("A12:I12"));
  styleSection(sheet.getRange("A20:I20"));
  styleHeader(sheet.getRange("A13:H13"));
  styleHeader(sheet.getRange("A21:I21"));
  sheet.getRange("A:A").format.columnWidth = 22;
  sheet.getRange("B:B").format.columnWidth = 48;
  sheet.getRange("C:C").format.columnWidth = 15;
  sheet.getRange("D:I").format.columnWidth = 18;
  sheet.freezePanes.freezeRows(3);
  return sheet;
}

async function main() {
  if (process.argv.length < 4) {
    throw new Error(
      "usage: node export_evaluation3d_xlsx.mjs evaluation3d.json evaluation3d.xlsx [preview_dir]"
    );
  }
  const jsonPath = path.resolve(process.argv[2]);
  const xlsxPath = path.resolve(process.argv[3]);
  const previewDir = process.argv[4] ? path.resolve(process.argv[4]) : null;
  const report = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

  const workbook = Workbook.create();
  workbook.worksheets.add("验收汇总");
  workbook.worksheets.add("逐点误差");
  workbook.worksheets.add("相机参与明细");
  workbook.worksheets.add("异常与缺失");
  workbook.worksheets.add("联合BA优化收敛情况");
  writeSummarySheet(workbook, report);
  writePointSheet(workbook, report);
  writeCameraParticipationSheet(workbook, report);
  writeIssueSheet(workbook, report);
  writeBundleAdjustmentSheet(workbook, report);

  const errors = workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  if (errors?.matches?.length) {
    throw new Error(`formula errors detected: ${JSON.stringify(errors)}`);
  }

  if (previewDir) {
    fs.mkdirSync(previewDir, { recursive: true });
    for (const sheetName of [
      "验收汇总", "逐点误差", "相机参与明细", "异常与缺失",
      "联合BA优化收敛情况",
    ]) {
      const render = await workbook.render({
        sheetName, autoCrop: "all", scale: 1, format: "png",
      });
      await fs.promises.writeFile(
        path.join(previewDir, `${sheetName}.png`),
        new Uint8Array(await render.arrayBuffer())
      );
    }
    const inspection = workbook.inspect({
      kind: "table",
      range: "验收汇总!A1:F36",
      include: "values,formulas",
      tableMaxRows: 40,
      tableMaxCols: 8,
      summary: "evaluation workbook summary inspection",
    });
    process.stdout.write(`${JSON.stringify(inspection, null, 2)}\n`);
  }

  fs.mkdirSync(path.dirname(xlsxPath), { recursive: true });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(xlsxPath);
}

await main();
