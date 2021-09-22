"""
Split entire dataset (CSV file) into training, testing and validation sets
Uses stratified random sampling for CNN inputs

Author: Venkanna Babu Guthula
Date: 22-09-2021
Email: g.venkanna37@gmail.com
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def split_data(args):
    csv_file = args.input_csv
    data = np.loadtxt(csv_file, delimiter=',', dtype='str')
    X = data[:, 0]
    y = data[:, 1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.175, random_state=42)

    # Test dataset
    X_test, y_test = np.expand_dims(X_test, axis=-1), np.expand_dims(y_test, axis=-1)
    test_data = np.hstack((X_test, y_test))
    filename = os.path.join(args.output_dir, args.file_name + "_test.csv")
    np.savetxt(filename, test_data, delimiter=",", fmt='%s')
    # Valid dataset
    X_valid, y_valid = np.expand_dims(X_valid, axis=-1), np.expand_dims(y_valid, axis=-1)
    valid_data = np.hstack((X_valid, y_valid))
    filename = os.path.join(args.output_dir, args.file_name + "_valid.csv")
    np.savetxt(filename, valid_data, delimiter=",", fmt='%s')
    # Training dataset
    X_train, y_train = np.expand_dims(X_train, axis=-1), np.expand_dims(y_train, axis=-1)
    train_data = np.hstack((X_train, y_train))
    filename = os.path.join(args.output_dir, args.file_name + "_train.csv")
    np.savetxt(filename,  train_data, delimiter=",", fmt='%s')
    print("Train, Test and Validation files created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="Image folder")
    parser.add_argument("--output_dir", type=str, help="Output directory where train, test and validation datasets save"
                        , default="../paths")
    parser.add_argument("--file_name", type=str, help="At the end of this file name, train, test, valid add for each"
                                                      " set")
    # parser.add_argument("--per_test_valid", type=int, help="Percentage of test and valid set", default="15")   # This has to change
    # parser.add_argument("--net_type", type=str, help="Architecture type CNN or FCN", default="fcn")
    args = parser.parse_args()
    split_data(args)
