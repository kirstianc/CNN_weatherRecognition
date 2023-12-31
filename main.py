# coding: utf-8
# NAME: main.py
"""
AUTHOR: Ian Chavez

Unpublished-rights reserved under the copyright laws of the United States.

This data and information is proprietary to, and a valuable trade secret of Ian Chavez. It is given in confidence by
Ian Chavez. Its use, duplication, or disclosure is subject to the restrictions set forth in the License Agreement under which it has been
distributed.

Unpublished Copyright © 2023 Ian Chavez

All Rights Reserved
"""
import os
import shutil
import image_class_map
import preprocess
import train_cnn
import test_cnn


def main():
    print("---- Starting Main ----")

    print("Cleaning up...")
    # delete processed_dataset folder if it exists --> clean start every time
    if os.path.exists("processed_dataset"):
        shutil.rmtree("processed_dataset")

    print("Creating image_class_map.csv...")
    image_class_map.main()

    print("Preprocessing...")
    train_loader, valid_loader, test_loader = preprocess.main()

    print("Training...")
    train_cnn.main(train_loader, valid_loader)

    print("Testing...")
    test_cnn.test_cnn(test_loader)


if __name__ == "__main__":
    main()
