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
  workbook.worksheets.add("异常与缺失");
  writeSummarySheet(workbook, report);
  writePointSheet(workbook, report);
  writeIssueSheet(workbook, report);

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
    for (const sheetName of ["验收汇总", "逐点误差", "异常与缺失"]) {
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
