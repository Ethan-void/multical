import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const COLORS = {
  navy: "#D9D9D9",
  blue: "#D9D9D9",
  paleBlue: "#FFFFFF",
  paleGreen: "#FFFFFF",
  paleOrange: "#FFFFFF",
  paleYellow: "#FFFFFF",
  paleRed: "#FFFFFF",
  text: "#000000",
  border: "#BFBFBF",
  white: "#FFFFFF",
};

function setBorders(range) {
  range.format.borders = {
    bottom: { style: "thin", color: COLORS.border },
    left: { style: "thin", color: COLORS.border },
    right: { style: "thin", color: COLORS.border },
    insideHorizontal: { style: "thin", color: COLORS.border },
    insideVertical: { style: "thin", color: COLORS.border },
  };
}

function styleTitle(range) {
  range.format = {
    fill: COLORS.navy,
    font: { name: "Aptos Display", bold: true, color: COLORS.text, size: 16 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.blue,
    font: { name: "Aptos", bold: true, color: COLORS.text, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 30;
  setBorders(range);
}

function styleSection(range) {
  range.format = {
    fill: COLORS.white,
    font: { name: "Aptos", bold: true, color: COLORS.text, size: 11 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 22;
}

function styleBody(range) {
  range.format.font = { name: "Aptos", size: 10 };
  range.format.verticalAlignment = "center";
  setBorders(range);
}

function writeSource(sheet, row, lastColumn, source, label = "源文件") {
  sheet.getRange(`A${row}`).values = [[label]];
  sheet.mergeCells(`B${row}:${lastColumn}${row}`);
  sheet.getRange(`B${row}`).values = [[source ?? "未找到"]];
  sheet.getRange(`A${row}`).format.font = { bold: true, color: COLORS.text };
  sheet.getRange(`B${row}`).format.wrapText = true;
}

function percentRange(minimum, maximum) {
  if (minimum === null || maximum === null) return null;
  return `${(minimum * 100).toFixed(1)}%–${(maximum * 100).toFixed(1)}%`;
}

function writeIntrinsicSheet(workbook, report) {
  const sheet = workbook.worksheets.add("内参汇总");
  sheet.showGridLines = false;
  const headers = [
    "相机",
    "整体 RMS（px）",
    "平均视图 RMS（px）",
    "最差视图 RMS（px）",
    "最终视图",
    "最差视图",
    "水平覆盖区间",
    "垂直覆盖区间",
    "覆盖宽度",
    "覆盖高度",
  ];
  sheet.getRange("A1:J1").values = [headers];
  styleHeader(sheet.getRange("A1:J1"));

  const rows = report.rows.map((row) => [
    row.camera,
    row.RMS,
    row.mean_view_RMS,
    row.max_view_RMS,
    row.view_count,
    row.worst_view,
    percentRange(row.x_min_rate, row.x_max_rate),
    percentRange(row.y_min_rate, row.y_max_rate),
    row.width_rate,
    row.height_rate,
  ]);
  if (rows.length) {
    const lastRow = rows.length + 1;
    sheet.getRange(`A2:J${lastRow}`).values = rows;
    styleBody(sheet.getRange(`A2:J${lastRow}`));
    sheet.getRange(`B2:D${lastRow}`).format.numberFormat = "0.000";
    sheet.getRange(`E2:E${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`I2:J${lastRow}`).format.numberFormat = "0.0%";
  }

  if (!report.has_quality) {
    const noticeRow = rows.length + 3;
    sheet.mergeCells(`A${noticeRow}:J${noticeRow}`);
    sheet.getRange(`A${noticeRow}`).values = [[
      "输入文件未包含 quality 字段；请使用支持质量导出的内参文件。",
    ]];
    sheet.getRange(`A${noticeRow}:J${noticeRow}`).format.fill =
      COLORS.paleOrange;
  }
  if (!report.has_coverage) {
    const noticeRow = rows.length + 4;
    sheet.mergeCells(`A${noticeRow}:J${noticeRow}`);
    sheet.getRange(`A${noticeRow}`).values = [[
      "未找到内参角点缓存，画面覆盖列留空。可用 --intrinsic-detections 指定 detections.pkl。",
    ]];
    sheet.getRange(`A${noticeRow}:J${noticeRow}`).format.fill =
      COLORS.paleOrange;
  }

  sheet.getRange("A:A").format.columnWidth = 14;
  sheet.getRange("B:D").format.columnWidth = 18;
  sheet.getRange("E:E").format.columnWidth = 14;
  sheet.getRange("F:F").format.columnWidth = 28;
  sheet.getRange("G:H").format.columnWidth = 18;
  sheet.getRange("I:J").format.columnWidth = 14;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}

function writeExtrinsicSheet(workbook, report) {
  const sheet = workbook.worksheets.add("外参汇总");
  sheet.showGridLines = false;
  sheet.mergeCells("A1:L1");
  sheet.getRange("A1").values = [["各相机观测质量"]];
  styleSection(sheet.getRange("A1:L1"));
  sheet.getRange("A2:J2").values = [[
    "相机",
    "总观测数",
    "内点数",
    "异常点数",
    "内点率",
    "检测帧",
    "内点帧",
    "整帧剔除",
    "内点 RMS（px）",
    "全部点 RMS（px）",
  ]];
  styleHeader(sheet.getRange("A2:J2"));
  const cameraRows = (report.extrinsic_quality?.cameras ?? []).map((row) => [
    row.camera,
    row.observation_count,
    row.inlier_count,
    row.outlier_count,
    row.inlier_ratio,
    row.detected_frame_count,
    row.inlier_frame_count,
    row.rejected_frame_count,
    row.reprojection_RMS_px,
    row.all_points_RMS_px,
  ]);
  if (cameraRows.length) {
    const lastRow = 2 + cameraRows.length;
    sheet.getRange(`A3:J${lastRow}`).values = cameraRows;
    styleBody(sheet.getRange(`A3:J${lastRow}`));
    sheet.getRange(`B3:D${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`E3:E${lastRow}`).format.numberFormat = "0.00%";
    sheet.getRange(`F3:H${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`I3:J${lastRow}`).format.numberFormat = "0.000";
  } else {
    sheet.mergeCells("A3:J4");
    sheet.getRange("A3").values = [[
      "没有可用的逐相机观测质量数据。",
    ]];
    sheet.getRange("A3:J4").format = {
      fill: COLORS.paleOrange,
      wrapText: true,
      verticalAlignment: "center",
    };
  }

  sheet.getRange("A:A").format.columnWidth = 16;
  sheet.getRange("B:H").format.columnWidth = 14;
  sheet.getRange("I:J").format.columnWidth = 18;
  if (report.extrinsic_quality?.available) {
    appendExtrinsicPairQuality(
      sheet,
      report,
      4 + cameraRows.length
    );
  }
  sheet.freezePanes.freezeRows(2);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}

function writeExtrinsicCameraQualitySheet(workbook, report) {
  const quality = report.extrinsic_quality;
  const sheet = workbook.worksheets.add("外参相机质量");
  sheet.showGridLines = false;
  sheet.mergeCells("A1:J1");
  sheet.getRange("A1").values = [["外参诊断：按相机质量"]];
  styleTitle(sheet.getRange("A1:J1"));
  writeSource(sheet, 2, "J", report.source);
  sheet.mergeCells("A3:J3");
  sheet.getRange("A3").values = [[
    "诊断字段不参与优化或异常点剔除；RMS 单位为像素。",
  ]];
  sheet.getRange("A3:J3").format = {
    fill: COLORS.paleBlue,
    font: { name: "Aptos", color: COLORS.text, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };

  sheet.getRange("A5:J5").values = [[
    "相机",
    "总观测数",
    "内点数",
    "异常点数",
    "内点率",
    "检测帧",
    "内点帧",
    "整帧剔除",
    "内点 RMS（px）",
    "全部点 RMS（px）",
  ]];
  styleHeader(sheet.getRange("A5:J5"));
  const rows = quality.cameras.map((row) => [
    row.camera,
    row.observation_count,
    row.inlier_count,
    row.outlier_count,
    row.inlier_ratio,
    row.detected_frame_count,
    row.inlier_frame_count,
    row.rejected_frame_count,
    row.reprojection_RMS_px,
    row.all_points_RMS_px,
  ]);
  if (rows.length) {
    const lastRow = 5 + rows.length;
    sheet.getRange(`A6:J${lastRow}`).values = rows;
    styleBody(sheet.getRange(`A6:J${lastRow}`));
    sheet.getRange(`B6:D${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`E6:E${lastRow}`).format.numberFormat = "0.00%";
    sheet.getRange(`F6:H${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`I6:J${lastRow}`).format.numberFormat = "0.000";
    sheet.getRange(`E6:E${lastRow}`)
      .conditionalFormats.add("colorScale", {
        colors: [COLORS.paleRed, COLORS.paleYellow, COLORS.paleGreen],
        thresholds: ["min", "50%", "max"],
      });
    sheet.getRange(`H6:H${lastRow}`)
      .conditionalFormats.add("colorScale", {
        colors: [COLORS.paleGreen, COLORS.paleYellow, COLORS.paleRed],
        thresholds: ["min", "50%", "max"],
      });
    sheet.getRange(`I6:J${lastRow}`)
      .conditionalFormats.add("colorScale", {
        colors: [COLORS.paleGreen, COLORS.paleYellow, COLORS.paleRed],
        thresholds: ["min", "50%", "max"],
      });
  }

  sheet.getRange("A:A").format.columnWidth = 14;
  sheet.getRange("B:H").format.columnWidth = 14;
  sheet.getRange("I:J").format.columnWidth = 18;
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}

function appendExtrinsicPairQuality(sheet, report, sectionRow) {
  const quality = report.extrinsic_quality;
  const headerRow = sectionRow + 1;
  const firstDataRow = headerRow + 1;
  sheet.mergeCells(`A${sectionRow}:L${sectionRow}`);
  sheet.getRange(`A${sectionRow}`).values = [["外参相机对质量"]];
  styleSection(sheet.getRange(`A${sectionRow}:L${sectionRow}`));
  sheet.getRange(`A${headerRow}:L${headerRow}`).values = [[
    "相机对",
    "共同帧",
    "共同角点",
    "共同内点",
    "相机对 RMS（px）",
    "共同位姿",
    "位姿内点",
    "旋转离散度（°）",
    "平移离散度",
    "相对最终边旋转 RMS（°）",
    "相对最终边平移 RMS",
    "状态",
  ]];
  styleHeader(sheet.getRange(`A${headerRow}:L${headerRow}`));
  const rows = quality.pairs.map((row) => [
    row.pair,
    row.common_frame_count,
    row.common_corner_count,
    row.common_inlier_corner_count,
    row.pair_reprojection_RMS_px,
    row.common_pose_count,
    row.pose_inlier_count,
    row.rotation_scatter_deg,
    row.translation_scatter,
    row.final_rotation_residual_rms_deg,
    row.final_translation_residual_rms,
    row.status,
  ]);
  if (rows.length) {
    const lastRow = headerRow + rows.length;
    sheet.getRange(`A${firstDataRow}:L${lastRow}`).values = rows;
    styleBody(sheet.getRange(`A${firstDataRow}:L${lastRow}`));
    sheet.getRange(`B${firstDataRow}:D${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`E${firstDataRow}:E${lastRow}`).format.numberFormat = "0.000";
    sheet.getRange(`F${firstDataRow}:G${lastRow}`).format.numberFormat = "#,##0";
    sheet.getRange(`H${firstDataRow}:K${lastRow}`).format.numberFormat = "0.0000";
  }

  const thresholdRow = headerRow + rows.length + 2;
  sheet.mergeCells(`A${thresholdRow}:L${thresholdRow}`);
  sheet.getRange(`A${thresholdRow}`).values = [["状态阈值"]];
  styleSection(sheet.getRange(`A${thresholdRow}:L${thresholdRow}`));
  sheet.getRange(`A${thresholdRow + 1}:D${thresholdRow + 1}`).values = [[
    "状态", "最少共同帧", "最大旋转离散度（°）", "最大平移离散度",
  ]];
  styleHeader(sheet.getRange(`A${thresholdRow + 1}:D${thresholdRow + 1}`));
  const thresholdRows = Object.entries(quality.thresholds).map(
    ([status, values]) => [
      status,
      values.min_common_frames,
      values.max_rotation_scatter_deg,
      values.max_translation_scatter,
    ]
  );
  if (thresholdRows.length) {
    const first = thresholdRow + 2;
    const last = first + thresholdRows.length - 1;
    sheet.getRange(`A${first}:D${last}`).values = thresholdRows;
    styleBody(sheet.getRange(`A${first}:D${last}`));
    sheet.getRange(`B${first}:B${last}`).format.numberFormat = "#,##0";
    sheet.getRange(`C${first}:D${last}`).format.numberFormat = "0.0000";
  }

  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange("B:D").format.columnWidth = 14;
  sheet.getRange("E:E").format.columnWidth = 18;
  sheet.getRange("F:G").format.columnWidth = 14;
  sheet.getRange("H:I").format.columnWidth = 18;
  sheet.getRange("J:K").format.columnWidth = 22;
  sheet.getRange("L:L").format.columnWidth = 14;
}

function writeExtrinsicGraphSheet(workbook, report) {
  const quality = report.extrinsic_quality;
  const graph = quality.graph;
  const sheet = workbook.worksheets.add("外参图诊断");
  sheet.showGridLines = false;
  sheet.mergeCells("A1:H1");
  sheet.getRange("A1").values = [["外参连接图、闭环与警告"]];
  styleTitle(sheet.getRange("A1:H1"));
  writeSource(sheet, 2, "H", report.source);

  sheet.mergeCells("A4:H4");
  sheet.getRange("A4").values = [["连接图"]];
  styleSection(sheet.getRange("A4:H4"));
  sheet.getRange("A5:B9").values = [
    ["图已连通", graph.connected === true ? "是" : graph.connected === false ? "否" : null],
    ["初始化主相机", graph.initialization_master],
    ["初始化树边", graph.selected_initialization_edges.join(", ")],
    ["弱初始化树边", graph.weak_selected_edges.join(", ")],
    ["诊断字段", quality.diagnostic_only ? "只诊断，不参与优化" : "未声明"],
  ];
  styleBody(sheet.getRange("A5:B9"));
  sheet.getRange("A5:A9").format.font = {
    name: "Aptos", bold: true, color: COLORS.text, size: 10,
  };
  sheet.getRange("B7:B8").format.wrapText = true;

  sheet.mergeCells("A11:H11");
  sheet.getRange("A11").values = [["冗余边闭环一致性"]];
  styleSection(sheet.getRange("A11:H11"));
  sheet.getRange("A12:D12").values = [[
    "相机对",
    "旋转闭环误差（°）",
    "平移闭环误差",
    "Frobenius 闭环误差",
  ]];
  styleHeader(sheet.getRange("A12:D12"));
  const closureRows = graph.redundant_edge_consistency.map((row) => [
    row.pair,
    row.rotation_closure_error_deg,
    row.translation_closure_error,
    row.frobenius_closure_error,
  ]);
  let closureLastRow = 12;
  if (closureRows.length) {
    closureLastRow += closureRows.length;
    sheet.getRange(`A13:D${closureLastRow}`).values = closureRows;
    styleBody(sheet.getRange(`A13:D${closureLastRow}`));
    sheet.getRange(`B13:D${closureLastRow}`).format.numberFormat = "0.0000";
    sheet.getRange(`A13:D${closureLastRow}`).format.fill =
      COLORS.paleOrange;
  }

  const warningSectionRow = closureLastRow + 2;
  const warningHeaderRow = warningSectionRow + 1;
  sheet.mergeCells(`A${warningSectionRow}:H${warningSectionRow}`);
  sheet.getRange(`A${warningSectionRow}`).values = [["诊断警告"]];
  styleSection(sheet.getRange(`A${warningSectionRow}:H${warningSectionRow}`));
  sheet.getRange(`A${warningHeaderRow}:D${warningHeaderRow}`).values = [[
    "代码", "相机", "相机对", "说明",
  ]];
  styleHeader(sheet.getRange(`A${warningHeaderRow}:D${warningHeaderRow}`));
  const warningRows = quality.warnings.map((row) => [
    row.code, row.camera, row.pair, row.message,
  ]);
  if (warningRows.length) {
    const warningLastRow = warningHeaderRow + warningRows.length;
    sheet.getRange(`A${warningHeaderRow + 1}:D${warningLastRow}`).values =
      warningRows;
    styleBody(sheet.getRange(`A${warningHeaderRow + 1}:D${warningLastRow}`));
    sheet.getRange(`A${warningHeaderRow + 1}:D${warningLastRow}`)
      .format.fill = COLORS.paleOrange;
    sheet.getRange(`D${warningHeaderRow + 1}:D${warningLastRow}`)
      .format.wrapText = true;
  }

  sheet.getRange("A:A").format.columnWidth = 28;
  sheet.getRange("B:C").format.columnWidth = 22;
  sheet.getRange("D:D").format.columnWidth = 72;
  sheet.getRange("E:H").format.columnWidth = 14;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

async function main() {
  if (process.argv.length < 4) {
    throw new Error(
      "usage: node export_calibration_analysis_xlsx.mjs report.json output.xlsx [preview_dir]"
    );
  }
  const reportPath = path.resolve(process.argv[2]);
  const outputPath = path.resolve(process.argv[3]);
  const previewDir = process.argv[4] ? path.resolve(process.argv[4]) : null;
  const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));

  const workbook = Workbook.create();
  const sheetNames = [];
  if (report.intrinsic) {
    writeIntrinsicSheet(workbook, report.intrinsic);
    sheetNames.push("内参汇总");
  }
  if (report.extrinsic) {
    writeExtrinsicSheet(workbook, report.extrinsic);
    sheetNames.push("外参汇总");
  }
  if (!sheetNames.length) {
    throw new Error("report contains neither intrinsic nor extrinsic data");
  }

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
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);

  const savedWorkbook = await SpreadsheetFile.importXlsx(
    await FileBlob.load(outputPath)
  );
  const savedErrors = await savedWorkbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "saved workbook formula error scan",
  });
  if (savedErrors?.matches?.length) {
    throw new Error(
      `formula errors detected after export: ${JSON.stringify(savedErrors)}`
    );
  }

  if (previewDir) {
    fs.mkdirSync(previewDir, { recursive: true });
    for (const sheetName of sheetNames) {
      const render = await savedWorkbook.render({
        sheetName,
        autoCrop: "all",
        scale: 1.5,
        format: "png",
      });
      await fs.promises.writeFile(
        path.join(previewDir, `${sheetName}.png`),
        new Uint8Array(await render.arrayBuffer())
      );
    }
    for (const sheetName of sheetNames) {
      const inspection = await savedWorkbook.inspect({
        kind: "table",
        range: sheetName === "内参汇总"
          ? "内参汇总!A1:J15"
          : sheetName === "外参汇总"
            ? "外参汇总!A1:L60"
            : `${sheetName}!A1:L60`,
        include: "values,formulas",
        tableMaxRows: 60,
        tableMaxCols: 12,
        summary: `${sheetName} workbook inspection`,
      });
      process.stdout.write(
        `${inspection.ndjson ?? JSON.stringify(inspection)}\n`
      );
    }
  }
}

await main();
