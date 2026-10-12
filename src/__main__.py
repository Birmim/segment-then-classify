import os
import argparse
import data_prep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Folder Path")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        parser.error(f"Folder not found: {args.folder_path}")

    X_train, X_val, X_test, y_cnn_train, y_cnn_val, y_cnn_test, y_unet_train, y_unet_val, y_unet_test = data_prep.split_data(args.folder_path)