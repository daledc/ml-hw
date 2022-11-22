import pandas as pd
import numpy as np
import argparse

def generate_dataset(args):
    """
    
    """
    func = np.poly1d(args.coeff)
    print("function:\n", func)
    print("domain:", args.domain)
    rng = np.random.default_rng()
    a, b = args.domain
    x = (b-a)*rng.random((args.size,), dtype=np.float64) + a
    y = func(x)
    all_data = pd.DataFrame(data=zip(x, y), columns=["x", "y"])
    train_data = all_data.sample(frac=args.train_ratio, replace=False,
                                                        random_state=args.seed)
    test_data = all_data.drop(train_data.index)
    train_data.to_csv("data_train.csv", index = False)
    test_data.to_csv("data_test.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sz", "--size", default=1000, type=int,
                                    help="number of data points generated")
    parser.add_argument("-tr", "--train_ratio", default=0.8, type=float,
                                    help="training data percentage")
    parser.add_argument("-c", "--coeff", nargs="+", default=[1.0, 2.0, 3.0],
                        type=float, help="polynomial coefficients")
    parser.add_argument("-se", "--seed", default=1010, type=int,
                                    help="rng seed")
    parser.add_argument("-d", "--domain", nargs="+", default=[-10.0, 10.0],
                        type=float, help="input domain to be sampled")
    args = parser.parse_args()
    generate_dataset(args)

