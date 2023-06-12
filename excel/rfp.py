#!/usr/bin/env python3
import openpyxl
from openpyxl.utils import get_column_letter
import xlrd
import os
import glob
import argparse
from os.path import exists, isdir

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument('-dir', type=str, required=True)
    args = psr.parse_args()

    if not exists(args.dir):
        exit(f"Directory \"{args.dir}\" does not exist")
    
    if not isdir(args.dir):
        exit(f"{args.dir} is not a directory") 

    # every file that ends with '.xlsx' under "sheets" dir
    files = glob.glob(args.dir + '/**/*.xls*', recursive=True)

    print(files)
    for f in files:
        try:
            print(f"\n{f}")
            if f.endswith(".xlsx"):
                wb = openpyxl.load_workbook(f)
                sheet = wb[wb.sheetnames[0]]
                field_start = get_column_letter(sheet.min_column) + str(sheet.min_row)
                field_end = get_column_letter(sheet.max_column) + str(sheet.max_row)
                print(field_start, '--', field_end)
                print(sheet[field_start:field_end])

            elif f.endswith(".xls"):
                wb = xlrd.open_workbook(f)
                sheet = wb.sheet_by_index(0)
                # TODO
            
            
        except Exception as e:
            print(f"{e}")