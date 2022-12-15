"""This program learns a polynomial function (degree 1 to 4) using stochastic
gradient descent (SGD).

Typical usage example:
  python solver.py
"""
import os
import argparse
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import pandas as pd
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from tqdm import tqdm
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt




def load_dataset(path, x_name="x", y_name="y"):
    """
    Loads a csv file containing named columns into arrays X and Y.
    """
    data = pd.read_csv(path, index_col=False)
    X = data.get(x_name).values.reshape(-1, 1)
    Y = data.get(y_name).values.reshape(-1, 1)
    return X, Y


class DataLoader:
    """
    Defines a dataloader for loading a dataset into minibatches.
    """
    def __init__(self, X, Y, bs=8, input_transform=None, shuffle=False):
        assert len(X) == len(Y), ("Input and output must be of the same length.")
        self.X = X
        self.Y = Y
        self.bs = bs
        self.input_transform = input_transform
        self.shuffle = shuffle
        self.npts = len(X)

    def size(self):
        return self.npts

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()

        for start in range(0, self.size(), self.bs):
            xb = self.X[start:start+self.bs]
            if self.input_transform is not None:
                xb = self.input_transform(xb)
            yb = self.Y[start:start+self.bs]
            yield xb, yb

    def __len__(self):
        return self.size()//self.bs + (0 if self.size()%self.bs == 0 else 1)

    def shuffle_data(self):
        perm_index = np.random.permutation(self.size())
        self.X = self.X[perm_index]
        self.Y = self.Y[perm_index]


def split_dataset(X, Y):
    """
    Performs data augmentation and generates training and validation splits.
    """
    # Sort input dataset based on X
    perm_index = X.flatten().argsort()
    Y_sort = Y[perm_index]
    X_sort = X[perm_index]

    # Perform data augmentation
    X_interp = np.arange(-20, 20 + 0.025, 0.025)
    Y_interp = np.interp(X_interp, X_sort.flatten(), Y_sort.flatten())
    X_aug, Y_aug = X_interp.reshape(-1, 1), Y_interp.reshape(-1, 1)

    valid_splits = 5
    parts = [np.arange(i, X_aug.shape[0],valid_splits) for i in range(valid_splits)]
    valid_idx = 0
    train_indices = np.sort(np.concatenate(parts[:valid_idx]+parts[valid_idx+1:]))
    valid_indices = np.concatenate(parts[valid_idx:valid_idx+1])
    X_train = X_aug[train_indices]
    Y_train = Y_aug[train_indices]
    X_valid = X_aug[valid_indices]
    Y_valid = Y_aug[valid_indices]
    return X_train, Y_train, X_valid, Y_valid


def TensorListToTensor(tensor_list):
    """
    Convert a list of tinygrad Tensor/s into a single tinygrad Tensor.
    """
    assert len(tensor_list) > 0, ("Cannot create empty Tensor.")
    T = tensor_list[0]
    for idx in range(1, len(tensor_list)):
        T = T.cat(tensor_list[idx])
    return T


class PolyNet:
    """
    A polynomial model for tinygrad.
    """
    def __init__(self, degree=None):
        if degree is not None:
            self.coeff_tl = [Tensor([np.random.normal(0, 1, 1)],
                                requires_grad=True) for idx in range(degree+1)]
        else:
            self.coeff_tl = None

    def forward(self, x):
        assert self.coeff_tl is not None
        theta = TensorListToTensor(self.coeff_tl)
        out = x.matmul(theta).sum(axis=1)
        return out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.coeff_tl

    def degree(self):
        return len(self.coeff_tl)-1

    def coefficients(self):
        return TensorListToTensor(self.coeff_tl).numpy().flatten()

    def load_coefficients(self, coeff_list, requires_grad=False):
        self.coeff_tl = []
        for coeff in coeff_list:
            self.coeff_tl.append(Tensor([[coeff]], requires_grad=requires_grad))


class SGD(optim.Optimizer):
    """
    Modified SGD implementation of tinygrad with support for multiple learning rates
    and momentum.
    """
    def __init__(self, params, lr_list, beta=0.9):
        super().__init__(params)
        self.prev_grad = [0.0]*len(self.params)
        self.lr_list = lr_list
        if type(self.lr_list) is not list:
            self.lr_list = [self.lr_list]*len(self.params)
        self.beta = beta

    def step(self):
        for idx, t, lr in zip(range(len(self.params)), self.params, self.lr_list):
            new_grad = self.beta*self.prev_grad[idx] + (1-self.beta)*t.grad
            t.assign(t.detach() - new_grad * lr)
            self.prev_grad[idx] = new_grad
        self.realize()


class PolynomialTransform:
    """
    Extend the dimension of an array of scalars by computing the nth power of
    each element from 0 to the specified degree in an additional axis.
    """
    def __init__(self, degree):
        assert degree > 0, "Degree must be greater than 0."
        self.degree = degree

    def __call__(self, X):
        X = Tensor(X, requires_grad=False)
        out = X.pow(0)
        for exp in range(1, self.degree+1):
            out = out.cat(X.pow(exp), dim=1)
        return out


class EarlyStopping:
    """
    Tracks the model validation loss to perform early stopping. The model
    weights corresponding to the lowest validation loss is stored.
    """
    def __init__(self, epochs=5, min_loss=1e20):
        self.stop_epoch = epochs
        self.min_loss = min_loss
        self.epochs_since_min = 0
        self.best_coefficients = None

    def update(self, loss, model=None):
        if loss < self.min_loss:
            self.min_loss = loss
            self.epochs_since_min = 0
            if model is not None:
                self.best_coefficients = model.coefficients()
        else:
            self.epochs_since_min += 1

    def stop(self):
        if self.epochs_since_min >= self.stop_epoch:
            return True
        return False


class ModelSelection:
    """
    Tracks the model weights of the each possible model based on a model cost.
    """
    def __init__(self, degree=0, cost=1e20, verbose=True):
        self.best_degree = degree
        self.best_cost = cost
        self.best_coefficients = []
        self.verbose = verbose

    def update(self, model, model_cost, best_coefficients):
        if model_cost < self.best_cost:
            self.best_cost = model_cost
            self.best_degree = model.degree()
            self.best_coefficients = best_coefficients
        if self.verbose:
            print(f"Model Cost: {model_cost:.4f}, Coefficients: {best_coefficients}")


def MSEloss(y, yhat):
    """
    Computes the MSE between two tinygrad Tensors
    """
    mse_loss = (yhat-y).mul(yhat-y).mean()
    return mse_loss


def train_one_epoch(model, train_dataloader, optimizer, lambda_reg):
    """
    Trains the input model on the input dataloader; backpropagation is
    controlled by the input optimizer.
    """
    train_loss = 0.0
    for x, y in train_dataloader:
        optimizer.zero_grad()
        out = model.forward(x).reshape((-1,1))
        loss = MSEloss(y, out)
        W = TensorListToTensor(model.parameters())
        reg_loss = lambda_reg * W[1:].mul(W[1:]).sqrt().sum()
        train_batch_loss = loss + reg_loss
        train_loss += train_batch_loss*out.shape[0]
        train_batch_loss.backward()
        optimizer.step()

    train_loss /= train_dataloader.size()
    return model, train_loss


def evaluate_model(model, dataloader):
    """
    Computes the mean square error of the model on the given dataloader.
    """
    mse = 0.0
    for x, y in dataloader:
        out = model.forward(x).reshape((-1,1))
        loss = MSEloss(y, out).numpy()[0]
        mse += loss*out.shape[0]
    mse /= dataloader.size()
    return mse


def visualize_results(model, input_transform, X, Y, split, result_path, npts=100, save=True):
    """
    Plots the model polynomial against the datapoints (X,Y) provided.
    """
    X_disp = np.linspace(np.min(X), np.max(X), npts).reshape(-1, 1)
    Y_disp = model.forward(input_transform(X_disp)).reshape(-1, 1).numpy()
    plt.plot(X, Y, 'r.'), plt.plot(X_disp, Y_disp, 'b')
    plt.xlabel('x'), plt.ylabel('y')
    plt.title(f'Model Fit Visualization on {split.capitalize()} Set')
    plt.legend([f'{split.capitalize()} Set', 'Learned Model'])
    if save:
        plt.savefig(os.path.join(result_path, 
                                        f'model_fit_{split.lower()}.png'))
    plt.show()


def generate_predictions(model, input_transform, X, split, result_path, save=True):
    """
    Generates the prediction results of the model on the input data X.
    """
    Ypred = model.forward(input_transform(X)).reshape((-1,1)).numpy() 
    if save:
        preds = pd.DataFrame(data=zip(X.reshape(-1), Ypred.reshape(-1)),
                                columns=["x", "y"])
        preds.to_csv(os.path.join(result_path, f"predictions_{split.lower()}.csv"),
                                    index = False)
    return Ypred




if __name__ == "__main__":

    # Program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='.', type=str,
                                    help="path to data")
    parser.add_argument("-r", "--results", default='results', type=str,
                                    help="path to data")
    parser.add_argument("-e", "--epochs", default=100, type=int,
                                    help="maximum epochs per model")
    args = parser.parse_args()

    # Create results path if it does not exist
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    # Load dataset
    X, Y = load_dataset(os.path.join(args.data, "data_train.csv"))
    X_train, Y_train, X_valid, Y_valid = split_dataset(X, Y)
    X_test, Y_test = load_dataset(os.path.join(args.data, "data_test.csv"))

    # Initialize parameters that do not need to be updated for each model degree
    batch_size = 128
    lr_list = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]
    model_selector = ModelSelection(degree=0, cost=1e20)

    # Train polynomial models from degree 1 to degree 4
    for degree in range(1, 5):

        # Initialize model with current degree and other training objects that
        # change based on degree
        model = PolyNet(degree)
        optimizer = SGD(model.parameters(), lr_list=lr_list[:degree+1])
        early_stop = EarlyStopping(epochs=20)
        lambda_reg = 1/degree
        input_transform = PolynomialTransform(degree)

        # Train current polynomial model with current degree
        train_dataloader = DataLoader(X_train, Y_train, batch_size,
                                        input_transform, shuffle=True)
        valid_dataloader = DataLoader(X_valid, Y_valid, batch_size,
                                        input_transform)

        num_epochs_pbar = tqdm(range(args.epochs))
        for epoch in num_epochs_pbar:
            # Training step
            model, train_loss = train_one_epoch(model, train_dataloader,
                                                        optimizer, lambda_reg)

            # Validation step
            valid_loss = evaluate_model(model, valid_dataloader)

            # Update early stop Scheduler
            early_stop.update(valid_loss, model)
            if early_stop.stop():
                break

            # Update progress bar after every epoch
            num_epochs_pbar.set_description(f"Degree {degree}; \
                Train Loss: \{train_loss.numpy()[0]:.4f}; \
                Valid Loss: {valid_loss:.4f}; \
                Coefficients: {model.coefficients()}")

        # Model selection cost is set to loss and penalizes a leading coefficient close to 0.
        model_cost = valid_loss + 1/abs(early_stop.best_coefficients[-1])
        model_selector.update(model, model_cost, early_stop.best_coefficients)

    # Load and display coefficients of best model
    model = PolyNet()
    input_transform = PolynomialTransform(model_selector.best_degree)
    model_func = np.poly1d(model_selector.best_coefficients[::-1])
    print(f"\nLearned Polynomial Model:\n{model_func}")
    with open(os.path.join(args.results, 'model.txt'),'w+') as f:
        f.write(str(model_func))
    model.load_coefficients(model_selector.best_coefficients)

    # Compute MSE on test set
    X_test, Y_test = load_dataset(os.path.join(args.data, "data_test.csv"))
    test_dataloader = DataLoader(X_test, Y_test, batch_size, input_transform)
    mse = evaluate_model(model, test_dataloader)
    print(f"Test MSE: {mse:.4f}")

    # Visualize performance on training and test sets and save to results path
    visualize_results(model, input_transform, X, Y, "train", args.results)
    visualize_results(model, input_transform, X_test, Y_test, "test", args.results)

    # Generate model predictions of training and test sets and save to results path
    generate_predictions(model, input_transform, X, "train", args.results)
    generate_predictions(model, input_transform, X_test, "test", args.results)

    # Display output polynomial using alternative method for comparison
    output = polyfit(X.flatten(), Y.flatten(), deg=degree)
    print(f"\nNumpy Polyfit Model:\n{np.poly1d(output[::-1])}")
    model_polyfit = PolyNet(degree)
    model_polyfit.load_coefficients(output)
    mse = evaluate_model(model_polyfit, test_dataloader)
    print(f"Test MSE: {mse:.4f}")