"""
Split entire dataset (CSV file) into training, testing and validation sets
Use stratified random sampling

Author: Venkanna Babu Guthula
Date: 22-09-2021
Example:
python split_dataset.py --input_csv ../paths/data_rnb.csv --output_dir ../paths/ --file_name rnb_subset --test_per 0.15
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def split_data(csv_file, test_per, valid_per, output_dir, file_name):
    # csv_file = args.input_csv
    data = np.loadtxt(csv_file, delimiter=',', dtype='str')
    X = data[:, 0]
    y = data[:, 1]

    if valid_per is None:
        valid_per = valid_set(test_per)
    else:
        valid_per = valid_per / (1 - valid_per)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_per, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_per, random_state=42)

    # Test dataset
    X_test, y_test = np.expand_dims(X_test, axis=-1), np.expand_dims(y_test, axis=-1)
    test_data = np.hstack((X_test, y_test))
    filename = os.path.join(output_dir, file_name + "_test.csv")
    np.savetxt(filename, test_data, delimiter=",", fmt='%s')

    # Valid dataset
    X_valid, y_valid = np.expand_dims(X_valid, axis=-1), np.expand_dims(y_valid, axis=-1)
    valid_data = np.hstack((X_valid, y_valid))
    filename = os.path.join(output_dir, file_name + "_valid.csv")
    np.savetxt(filename, valid_data, delimiter=",", fmt='%s')

    # Training dataset
    X_train, y_train = np.expand_dims(X_train, axis=-1), np.expand_dims(y_train, axis=-1)
    train_data = np.hstack((X_train, y_train))
    filename = os.path.join(output_dir, file_name + "_train.csv")
    np.savetxt(filename, train_data, delimiter=",", fmt='%s')
    print("Train, Test and Validation files created")


def valid_set(test_per):
    # percentage should be in fractions
    return test_per / (1 - test_per)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="Image folder")
    parser.add_argument("--output_dir", type=str, help="Output directory where train, test and validation datasets save"
                        , default="../paths")
    parser.add_argument("--file_name", type=str, help="At the end of this file name, train, test, valid add for each"
                                                      " set")
    parser.add_argument("--test_per", type=float, help="Percentage of test set", default=0.15)
    parser.add_argument("--valid_per", type=float, help="Percentage of valid set. if not given, valid per is equal to"
                                                        " test", default=None)
    args = parser.parse_args()

    split_data(args.input_csv, args.test_per, args.valid_per, args.output_dir, args.file_name)