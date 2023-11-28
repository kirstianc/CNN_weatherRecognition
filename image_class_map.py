# coding: utf-8
# NAME: image_class_map.py
"""
AUTHOR: Ian Chavez

Unpublished-rights reserved under the copyright laws of the United States.

This data and information is proprietary to, and a valuable trade secret of Ian Chavez. It is given in confidence by
Ian Chavez. Its use, duplication, or disclosure is subject to the restrictions set forth in the License Agreement under which it has been
distributed.

Unpublished Copyright © 2023 Ian Chavez

All Rights Reserved
"""
"""
========================== MODIFICATION HISTORY ==============================
11/15/23:
MOD: Creation of file and initial function
AUTHOR: Ian Chavez
COMMENT: n/a
====================== END OF MODIFICATION HISTORY ============================
"""
import os
import csv


def main():
    print("---- Starting Image Class Map ----")
    print("Setting paths...")

    # Set paths
    data_dir = "./dataset/"
    csv_file_path = "image_class_map.csv"

    # If csv file exists, clear it
    if os.path.exists(csv_file_path):
        open(csv_file_path, "w").close()

    # Write to csv file, img name (i.e. 0001.jpg), path, and class (root folder)
    with open("image_class_map.csv", "w", newline="") as file:
        writer = csv.writer(file)
        field = ["Image", "Path", "Class"]
        writer.writerow(field)
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    path = os.path.join(root, file)
                    class_name = os.path.basename(root)
                    writer.writerow([file, path.replace(data_dir, ""), class_name])
