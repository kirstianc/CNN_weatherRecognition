import os
import shutil
import image_class_map
import preprocess
import train_cnn
import test_cnn


def main():
    print("---- Starting Main ----")
    print("Cleaning up...")
    # Delete the processed_dataset folder if it exists
    # if os.path.exists("processed_dataset"):
    #    shutil.rmtree("processed_dataset")

    # print("Creating image_class_map.csv...")
    # Run image_class_map.py
    # image_class_map.main()

    print("Preprocessing...")
    # Run preprocess.py and get the data loaders
    train_loader, valid_loader, test_loader = preprocess.main()

    # print("Training...")
    # Run train.py with the training and validation loaders
    # train_cnn.main(train_loader, valid_loader)

    print("Testing...")
    # Run test.py with the test loader
    test_cnn.test_cnn(test_loader)


if __name__ == "__main__":
    main()
