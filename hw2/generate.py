"""This program generates a dataset containing samples of a specified polynomial
function.

Typical usage example:
  python generate.py --data data --coefficients 0.1 -2 1 1 4 
"""

import pandas as pd
import numpy as np
import argparse
import os
rng = np.random.default_rng()


def generate_dataset(args):
    """
    Generate dataset of a polynomial function using the given program arguments.
    """
    # Create path for data if it does not exist
    if not os.path.exists(args.data):
        os.mkdir(args.data)

    # Display polynomial function generated
    func = np.poly1d(args.coefficients[::-1])
    print("function:\n", func)
    print("train arange:", args.train_range)
    print("test arange:", args.test_range)

    # Generate train data
    start, stop, skip = args.train_range
    x_train = np.arange(start, stop, skip)
    n_train = 2*rng.random((len(x_train),), dtype=np.float64) - 1
    y_train = func(x_train) + n_train
    train_data = pd.DataFrame(data=zip(x_train, y_train), columns=["x", "y"])
    train_data = train_data.round(decimals=1)
    train_data.to_csv(os.path.join(args.data, "data_train.csv"), index = False)

    # Generate test data
    start, stop, skip = args.test_range
    x_test = np.arange(start, stop, skip)
    n_test = 2*rng.random((len(x_test),), dtype=np.float64) - 1
    y_test = func(x_test) + n_test
    test_data = pd.DataFrame(data=zip(x_test, y_test), columns=["x", "y"])
    test_data = test_data.round(decimals=1)
    test_data.to_csv(os.path.join(args.data, "data_test.csv"), index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='.', type=str,
                                help="path to data")
    parser.add_argument("-c", "--coefficients", nargs="+", default=[-1., 2., -3., 4.],
            type=float, help="polynomial coefficients (highest coefficient first)")
    parser.add_argument("--train_range", nargs="+", default=[-20, 20.2, 0.2],
                        type=float, help="train data x-value np.arange() params")
    parser.add_argument("--test_range", nargs="+", default=[-100, 105, 5],
                        type=float, help="test data x-value np.arange() params")
    parser.add_argument("-s", "--seed", default=1010, type=int,
                                    help="rng seed")
    args = parser.parse_args()
    generate_dataset(args)