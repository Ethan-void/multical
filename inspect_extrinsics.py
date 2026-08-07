import openpyxl

path = "/Users/ethan/Downloads/外参汇总.xlsx"
workbook = openpyxl.load_workbook(path, data_only=False)
for sheet in workbook.worksheets:
    print("---", sheet.title)
    for row in sheet.iter_rows(values_only=False):
        values = [cell.value for cell in row]
        if any(value is not None for value in values):
            print(values)
