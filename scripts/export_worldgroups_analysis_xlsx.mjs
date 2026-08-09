import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const COLORS = {
  navy: "#1F4E78",
  blue: "#4472C4",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  paleRed: "#FCE4D6",
  border: "#D1D5DB",
  white: "#FFFFFF",
  text: "#172033",
};

function lightBorders(range) {
  range.format.borders = {
    bottom: { style: "thin", color: COLORS.border },
    insideHorizontal: { style: "thin", color: COLORS.border },
  };
}

function title(range) {
  range.format = {
    fill: COLORS.navy,
    font: { name: "Aptos Display", bold: true, color: COLORS.white, size: 16 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function section(range) {
  range.format = {
    fill: COLORS.paleBlue,
    font: { name: "Aptos", bold: true, color: COLORS.text, size: 11 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 22;
}

function header(range) {
  range.format = {
    fill: COLORS.blue,
    font: { name: "Aptos", bold: true, color: COLORS.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 32;
}

function body(range) {
  range.format.font = { name: "Aptos", color: COLORS.text, size: 10 };
  range.format.verticalAlignment = "center";
  lightBorders(range);
}

function writeSummary(workbook, report) {
  const sheet = workbook.worksheets.add("质量汇总");
  sheet.showGridLines = false;
  sheet.mergeCells("A1:F1");
  sheet.getRange("A1").values = [["分组世界控制网：四相机外参质量报告"]];
  title(sheet.getRange("A1:F1"));

  sheet.mergeCells("A3:B3");
  sheet.getRange("A3").values = [["总体结论"]];
  section(sheet.getRange("A3:B3"));
  sheet.getRange("A4:B13").values = [
    ["状态", report.acceptance.passed ? "通过" : "不通过"],
    ["相机数量", report.summary.camera_count],
    ["局部组数量", report.summary.group_count],
    ["全部相机对", report.summary.camera_pair_count],
    ["跨组相机对", report.summary.cross_group_pair_count],
    ["共享世界控制点", report.summary.shared_control_point_count],
    ["有效共享控制点", report.summary.valid_shared_control_point_count],
    ["共享点最差组均值（px）", report.summary.shared_control_worst_group_mean_px],
    ["共享点最大组间差（px）", report.summary.shared_control_max_group_spread_px],
    ["独立跨组 3D 点", report.summary.cross_group_3d_point_count],
  ];
  body(sheet.getRange("A4:B13"));
  sheet.getRange("A4:A13").format.font = { name: "Aptos", bold: true, size: 10 };
  sheet.getRange("B11:B12").format.numberFormat = "0.000";
  sheet.getRange("B4").format = {
    fill: report.acceptance.passed ? COLORS.paleGreen : COLORS.paleRed,
    font: {
      name: "Aptos", bold: true,
      color: report.acceptance.passed ? "#006100" : "#9C0006",
    },
    horizontalAlignment: "center",
  };

  sheet.mergeCells("A15:F15");
  sheet.getRange("A15").values = [["跨组 3D 独立验证"]];
  section(sheet.getRange("A15:F15"));
  sheet.getRange("A16:F16").values = [[
    "点数", "均值", "RMS", "中位数", "P95", "最大值",
  ]];
  header(sheet.getRange("A16:F16"));
  const stats = report.cross_group_3d.summary;
  sheet.getRange("A17:F17").values = [[
    stats.count, stats.mean, stats.RMS, stats.median, stats.p95, stats.max,
  ]];
  body(sheet.getRange("A17:F17"));
  sheet.getRange("B17:F17").format.numberFormat = "0.000000";

  sheet.mergeCells("A19:F19");
  sheet.getRange("A19").values = [["未通过原因"]];
  section(sheet.getRange("A19:F19"));
  const failures = report.acceptance.failures.length
    ? report.acceptance.failures : ["无"];
  failures.forEach((failure, index) => {
    const row = 20 + index;
    sheet.mergeCells(`A${row}:F${row}`);
    sheet.getRange(`A${row}`).values = [[failure]];
    sheet.getRange(`A${row}:F${row}`).format.wrapText = true;
  });

  const limitationRow = 21 + failures.length;
  sheet.mergeCells(`A${limitationRow}:F${limitationRow}`);
  sheet.getRange(`A${limitationRow}`).values = [["解释边界"]];
  section(sheet.getRange(`A${limitationRow}:F${limitationRow}`));
  report.limitations.forEach((value, index) => {
    const row = limitationRow + 1 + index;
    sheet.mergeCells(`A${row}:F${row}`);
    sheet.getRange(`A${row}`).values = [[value]];
    sheet.getRange(`A${row}:F${row}`).format.wrapText = true;
    sheet.getRange(`A${row}:F${row}`).format.rowHeight = 30;
  });

  sheet.getRange("A:A").format.columnWidth = 26;
  sheet.getRange("B:F").format.columnWidth = 18;
  sheet.freezePanes.freezeRows(1);
  return sheet;
}

function writeGroups(workbook, report) {
  const sheet = workbook.worksheets.add("分组质量");
  sheet.showGridLines = false;
  sheet.mergeCells("A1:F1");
  sheet.getRange("A1").values = [["局部双目标定质量"]];
  section(sheet.getRange("A1:F1"));
  sheet.getRange("A2:F2").values = [[
    "组", "相机", "局部外参 RMS（px）", "全部点 RMS（px）",
    "局部观测", "局部内点",
  ]];
  header(sheet.getRange("A2:F2"));
  const localRows = report.groups.map((group) => [
    group.name, group.cameras.join(", "),
    group.local_extrinsic_rms_px,
    group.local_extrinsic_all_points_rms_px,
    group.local_extrinsic_observation_count,
    group.local_extrinsic_inlier_count,
  ]);
  if (localRows.length) {
    const last = localRows.length + 2;
    sheet.getRange(`A3:F${last}`).values = localRows;
    body(sheet.getRange(`A3:F${last}`));
    sheet.getRange(`C3:D${last}`).format.numberFormat = "0.000";
    sheet.getRange(`E3:F${last}`).format.numberFormat = "#,##0";
  }

  const sectionRow = localRows.length + 5;
  sheet.mergeCells(`A${sectionRow}:J${sectionRow}`);
  sheet.getRange(`A${sectionRow}`).values = [["世界控制网锚定质量"]];
  section(sheet.getRange(`A${sectionRow}:J${sectionRow}`));
  const headerRow = sectionRow + 1;
  sheet.getRange(`A${headerRow}:J${headerRow}`).values = [[
    "组", "世界观测", "世界内点", "世界内点率", "内点 RMS（px）",
    "全部点 RMS（px）", "最大误差（px）", "旋转保持误差（°）",
    "平移保持误差", "优化成功",
  ]];
  header(sheet.getRange(`A${headerRow}:J${headerRow}`));
  const worldRows = report.groups.map((group) => [
    group.name,
    group.world_observation_count,
    group.world_inlier_count,
    group.world_inlier_rate,
    group.world_reprojection_rms_px,
    group.world_all_points_rms_px,
    group.world_all_points_max_px,
    group.local_consistency_rotation_deg,
    group.local_consistency_translation,
    group.optimization_success ? "是" : "否",
  ]);
  if (worldRows.length) {
    const first = headerRow + 1;
    const last = headerRow + worldRows.length;
    sheet.getRange(`A${first}:J${last}`).values = worldRows;
    body(sheet.getRange(`A${first}:J${last}`));
    sheet.getRange(`B${first}:C${last}`).format.numberFormat = "#,##0";
    sheet.getRange(`D${first}:D${last}`).format.numberFormat = "0.0%";
    sheet.getRange(`E${first}:I${last}`).format.numberFormat = "0.000000";
  }
  sheet.getRange("A:B").format.columnWidth = 18;
  sheet.getRange("C:J").format.columnWidth = 20;
  sheet.freezePanes.freezeRows(2);
  return sheet;
}

function writeCameras(workbook, report) {
  const sheet = workbook.worksheets.add("相机世界位姿");
  sheet.showGridLines = false;
  sheet.getRange("A1:E1").values = [[
    "相机", "组", `世界 X（${report.world_units}）`,
    `世界 Y（${report.world_units}）`, `世界 Z（${report.world_units}）`,
  ]];
  header(sheet.getRange("A1:E1"));
  const rows = report.cameras.map((camera) => [
    camera.camera, camera.group, ...camera.position_world,
  ]);
  if (rows.length) {
    const last = rows.length + 1;
    sheet.getRange(`A2:E${last}`).values = rows;
    body(sheet.getRange(`A2:E${last}`));
    sheet.getRange(`C2:E${last}`).format.numberFormat = "0.000000";
  }
  sheet.getRange("A:B").format.columnWidth = 18;
  sheet.getRange("C:E").format.columnWidth = 22;
  sheet.freezePanes.freezeRows(1);
  return sheet;
}

function writePairs(workbook, report) {
  const sheet = workbook.worksheets.add("相机对外参");
  sheet.showGridLines = false;
  sheet.getRange("A1:O1").values = [[
    "源相机", "目标相机", "源组", "目标组", "类型",
    `基线（${report.world_units}）`,
    `Tx（目标相机系，${report.world_units}）`,
    `Ty（目标相机系，${report.world_units}）`,
    `Tz（目标相机系，${report.world_units}）`,
    "旋转角（°）", "Roll（°）", "Pitch（°）", "Yaw（°）",
    "质量来源", "说明",
  ]];
  header(sheet.getRange("A1:O1"));
  const rows = report.camera_pairs.map((pair) => [
    pair.source_camera, pair.destination_camera,
    pair.source_group, pair.destination_group, pair.pair_type,
    pair.baseline, ...pair.translation_destination_frame,
    pair.rotation_angle_deg, pair.roll_deg, pair.pitch_deg, pair.yaw_deg,
    pair.pair_type === "cross_group" ? "世界控制网间接推导" : "局部双目标定",
    pair.pair_type === "cross_group"
      ? "质量由共享控制点和跨组 3D 验证支持" : "可结合本组 calibration.pkl 分析",
  ]);
  if (rows.length) {
    const last = rows.length + 1;
    sheet.getRange(`A2:O${last}`).values = rows;
    body(sheet.getRange(`A2:O${last}`));
    sheet.getRange(`F2:M${last}`).format.numberFormat = "0.000000";
    sheet.getRange(`N2:O${last}`).format.wrapText = true;
  }
  sheet.getRange("A:E").format.columnWidth = 16;
  sheet.getRange("F:M").format.columnWidth = 18;
  sheet.getRange("N:N").format.columnWidth = 24;
  sheet.getRange("O:O").format.columnWidth = 44;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  return sheet;
}

function writeSharedControls(workbook, report) {
  const sheet = workbook.worksheets.add("共享控制点");
  sheet.showGridLines = false;
  const groupNames = report.groups.map((group) => group.name);
  const headers = [
    "世界 X", "世界 Y", "世界 Z", "组均值差（px）", "最差组均值（px）",
  ];
  groupNames.forEach((name) => {
    headers.push(`${name} 观测`, `${name} 内点`, `${name} 均值（px）`, `${name} 最大（px）`);
  });
  const lastColumn = String.fromCharCode(64 + headers.length);
  sheet.getRange(`A1:${lastColumn}1`).values = [headers];
  header(sheet.getRange(`A1:${lastColumn}1`));
  const rows = report.shared_control_points.map((point) => {
    const row = [
      ...point.world_point,
      point.group_mean_spread_px,
      point.worst_group_mean_px,
    ];
    groupNames.forEach((name) => {
      const metrics = point.groups[name] ?? {};
      row.push(
        metrics.observation_count ?? null,
        metrics.inlier_count ?? null,
        metrics.mean_reprojection_error_px ?? null,
        metrics.max_reprojection_error_px ?? null,
      );
    });
    return row;
  });
  if (rows.length) {
    const last = rows.length + 1;
    sheet.getRange(`A2:${lastColumn}${last}`).values = rows;
    body(sheet.getRange(`A2:${lastColumn}${last}`));
    sheet.getRange(`A2:E${last}`).format.numberFormat = "0.000000";
  }
  sheet.getRange("A:C").format.columnWidth = 16;
  sheet.getRange(`D:${lastColumn}`).format.columnWidth = 18;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(3);
  return sheet;
}

function writeCross3d(workbook, report) {
  const sheet = workbook.worksheets.add("跨组3D验证");
  sheet.showGridLines = false;
  sheet.getRange("A1:I1").values = [[
    "点", "参与组", "参与相机", `3D 误差（${report.world_units}）`,
    "重投影 RMS（px）", "误差 X", "误差 Y", "误差 Z", "状态",
  ]];
  header(sheet.getRange("A1:I1"));
  const rows = report.cross_group_3d.points.map((point) => [
    point.frame, point.groups.join(", "), point.cameras_used.join(", "),
    point.error_3d, point.reprojection_rms_px,
    ...(point.error_xyz ?? [null, null, null]), "成功",
  ]);
  report.cross_group_3d.failed_reconstructions.forEach((point) => {
    rows.push([
      point.frame, point.groups.join(", "),
      point.cameras_available.join(", "), null, null, null, null, null,
      `失败：${point.reason}`,
    ]);
  });
  if (rows.length) {
    const last = rows.length + 1;
    sheet.getRange(`A2:I${last}`).values = rows;
    body(sheet.getRange(`A2:I${last}`));
    sheet.getRange(`D2:H${last}`).format.numberFormat = "0.000000";
    sheet.getRange(`I2:I${last}`).format.wrapText = true;
  }
  sheet.getRange("A:B").format.columnWidth = 18;
  sheet.getRange("C:C").format.columnWidth = 32;
  sheet.getRange("D:H").format.columnWidth = 18;
  sheet.getRange("I:I").format.columnWidth = 42;
  sheet.freezePanes.freezeRows(1);
  return sheet;
}

async function main() {
  if (process.argv.length < 4) {
    throw new Error(
      "usage: node export_worldgroups_analysis_xlsx.mjs report.json output.xlsx [preview_dir]",
    );
  }
  const reportPath = path.resolve(process.argv[2]);
  const outputPath = path.resolve(process.argv[3]);
  const previewDir = process.argv[4] ? path.resolve(process.argv[4]) : null;
  const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
  const workbook = Workbook.create();
  const sheets = [
    writeSummary(workbook, report),
    writeGroups(workbook, report),
    writeCameras(workbook, report),
    writePairs(workbook, report),
    writeSharedControls(workbook, report),
    writeCross3d(workbook, report),
  ];

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  if (errors?.matches?.length) {
    throw new Error(`formula errors detected: ${JSON.stringify(errors)}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const savedErrors = await saved.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "saved workbook formula error scan",
  });
  if (savedErrors?.matches?.length) {
    throw new Error(`saved workbook errors: ${JSON.stringify(savedErrors)}`);
  }

  if (previewDir) {
    fs.mkdirSync(previewDir, { recursive: true });
    for (const sheet of sheets) {
      const preview = await saved.render({
        sheetName: sheet.name, autoCrop: "all", scale: 1.25, format: "png",
      });
      await fs.promises.writeFile(
        path.join(previewDir, `${sheet.name}.png`),
        new Uint8Array(await preview.arrayBuffer()),
      );
      const inspection = await saved.inspect({
        kind: "table",
        range: `${sheet.name}!A1:O40`,
        include: "values,formulas",
        tableMaxRows: 40,
        tableMaxCols: 15,
        summary: `${sheet.name} inspection`,
      });
      process.stdout.write(`${inspection.ndjson ?? JSON.stringify(inspection)}\n`);
    }
  }
}

await main();
