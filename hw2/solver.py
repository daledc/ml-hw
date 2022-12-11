"""This program learns a polynomial function (degree 1 to 4) using stochastic
gradient descent (SGD).

Typical usage example:
  python solver.py
"""

from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
np.set_printoptions(precision=4, suppress=True)
DEBUG = False




def TensorListToTensor(tensor_list):
    """
    Convert a list of tinygrad Tensor to a single tinygrad Tensor.
    """
    assert len(tensor_list) > 0, ("Cannot create empty Tensor.")
    T = tensor_list[0]
    for idx in range(1, len(tensor_list)):
        T = T.cat(tensor_list[idx])
    return T


class PolyNet:
    """
    Tinygrad model for a polynomial.
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

    def coefficients(self):
        return TensorListToTensor(self.coeff_tl).numpy().flatten()

    def load_coefficients(self, coeff_list, requires_grad=False):
        self.coeff_tl = []
        for coeff in coeff_list:
            self.coeff_tl.append(Tensor([[coeff]], requires_grad=requires_grad))


class SGD(optim.Optimizer):
  """
  SGD implementation of tinygrad with support for multiple learning rates.
  """
  def __init__(self, params, lr_list):
    super().__init__(params)
    self.lr_list = lr_list

  def step(self):
    for lr, t in zip(self.lr_list, self.params):
      t.assign(t.detach() - t.grad * lr)
    self.realize()



class SGD_Mom(optim.Optimizer):
  """
  SGD implementation of tinygrad with support for multiple learning rates
  and momentum.
  """
  def __init__(self, params, lr_list, beta=0.9):
    super().__init__(params)
    self.lr_list = lr_list
    self.beta = beta
    self.prev_grad = []
    for _ in self.params:
      self.prev_grad.append(0.0)

  def step(self):
    for idx, t, lr in zip(range(len(self.params)), self.params, self.lr_list):
      new_grad = self.beta*self.prev_grad[idx] + (1-self.beta)*t.grad
      t.assign(t.detach() - new_grad * lr)
      self.prev_grad[idx] = new_grad
    self.realize()


def load_dataset(path):
    """
    Loads the dataset containing tuples.
    """
    data = pd.read_csv(path, index_col=False)
    X = data.get("x").values.reshape(-1, 1)
    Y = data.get("y").values.reshape(-1, 1)
    return X, Y


class DataLoader:
    """
    Defines a dataloader for loading the dataset into minibatches.
    """
    def __init__(self, X, Y, bs=8, input_transform=None, shuffle_data=False):
        assert len(X) == len(Y), ("Input and output must be of the same length.")
        self.num_pts = len(X)
        self.length = len(X)//bs + (0 if len(X)%bs == 0 else 1)
        self.X = X
        self.Y = Y
        self.bs = bs
        self.input_transform = input_transform
        self.shuffle_data = shuffle_data

    def __iter__(self):
        if self.shuffle_data:
            self.shuffle()

        for start in range(0, self.num_pts, self.bs):
            xb = self.X[start:start+self.bs]
            yb = self.Y[start:start+self.bs]
            if self.input_transform is not None:
                xb = self.input_transform(xb)
            yield xb, yb

    def __len__(self):
        return self.length

    def shuffle(self):
        perm_index = np.random.permutation(self.num_pts)
        self.X = self.X[perm_index]
        self.Y = self.Y[perm_index]


class polynomial_transform:
    """
    Converts a list of scalars into a polynomial evaluated tingrad Tensor.
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


class MSEloss:
    """
    Defines a class that computes the MSE between two tinygrad Tensors
    """
    def __init__(self):
        pass

    def __call__(self, y, yhat):
        mse_loss = (yhat-y).mul(yhat-y).mean()
        return mse_loss


class Scheduler:
    """
    Defines a class that tracks the lowest validation loss and corresponding
    model weights; and can perform early stopping.
    """
    def __init__(self, early_stop_epochs=5):
        self.min_loss = 1e30
        self.epochs_since_min = 0
        self.early_stop_epochs = early_stop_epochs
        self.best_coefficients = None

    def update(self, loss, model=None):
        if loss < self.min_loss:
            self.min_loss = loss
            self.epochs_since_min = 0
            if model is not None:
                self.best_coefficients = model.coefficients()
        else:
            self.epochs_since_min += 1

    def early_stop(self):
        if self.epochs_since_min >= self.early_stop_epochs:
            return True
        return False


def evaluate_model(model, dataloader):
    """
    Computes the mean square error of the model on the given dataloader.
    """
    mse = 0.0
    for x, y in dataloader:
        out = model.forward(x).reshape((-1,1))
        loss = loss_function(y, out).numpy()[0]
        mse += loss*out.shape[0]
    mse /= dataloader.num_pts
    return mse


def split_dataset(X, Y):
    """
    Performs data augmentation and generates training and validation splits.
    """

    # Sort input dataset based on X
    perm_index = X.flatten().argsort()
    Y_sort = Y[perm_index]
    X_sort = X[perm_index]

    # Perform data augmentation
    X_interp = np.arange(-20, 20 + 0.05, 0.05)
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


def train_one_epoch(model, train_dataloader, optimizer):
    """
    Trains the input model on the input dataloader; backpropagation is
    controlled by the input optimizer.
    """
    train_loss = 0.0
    for x, y in train_dataloader:
        optimizer.zero_grad()
        out = model.forward(x).reshape((-1,1))
        loss = loss_function(y, out)
        W = TensorListToTensor(model.parameters())
        l2_loss = lambda_l2 * W[1:].mul(W[1:]).sqrt().sum()
        train_batch_loss = loss + l2_loss
        train_loss += train_batch_loss*out.shape[0]
        train_batch_loss.backward()
        optimizer.step()

    train_loss /= train_dataloader.num_pts
    return model, train_loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='.', type=str,
                                    help="path to data")
    parser.add_argument("-e", "--epochs", default=100, type=int,
                                    help="maximum epochs per model")
    args = parser.parse_args()

    # Load data
    X, Y = load_dataset(os.path.join(args.data, "data_train.csv"))
    X_train, Y_train, X_valid, Y_valid = split_dataset(X, Y)
    X_test, Y_test = load_dataset(os.path.join(args.data, "data_test.csv"))

    # Create results path if it does not exist
    if not os.path.exists('./results'):
        os.mkdir('./results')

    if DEBUG:
        plt.plot(X_train, Y_train, 'ro')
        plt.plot(X_valid, Y_valid, 'b.')
        plt.legend(['Train Split', 'Valid Split'])
        plt.show()

    batch_size = 128
    init_lr_list=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9] # SGD_Mom
    # init_lr_list=[1e-1, 3e-4, 3e-6, 3e-8, 3e-10] # SGD
    num_epochs = args.epochs
    loss_function = MSEloss()
    best_degree = 0
    best_cost = 1e20
    best_coefficients = []

    for degree in range(4, 0, -1):
        # Initialize model with current degree
        print(f"\nDegree: {degree}")

        model = PolyNet(degree)
        optimizer = SGD_Mom(model.parameters(), lr_list=init_lr_list[:degree+1])
        # optimizer = optim.SGD(model.parameters(), lr=init_lr_list[degree])
        scheduler = Scheduler(early_stop_epochs=10)
        lambda_l2 = 1/(degree)
        input_transform = polynomial_transform(degree)

        if DEBUG:
            X_test, Y_test = load_dataset(os.path.join(args.data, "data_test.csv"))
            output = polyfit(X.flatten(), Y.flatten(), deg=degree)
            print("numpy polyfit output:", output)
            model2 = PolyNet(degree)
            model2.load_coefficients(output)
            test_dataloader = DataLoader(X_test, Y_test, batch_size, input_transform)
            mse = evaluate_model(model2, test_dataloader)
            print(f"Test MSE (polyfit degree={degree}): {mse:.4f}")

        train_dataloader = DataLoader(X_train, Y_train, batch_size, input_transform, shuffle_data=True)
        valid_dataloader = DataLoader(X_valid, Y_valid, batch_size, input_transform)

        num_epochs_pbar = tqdm(range(num_epochs))
        for epoch in num_epochs_pbar:
            # Training Step
            model, train_loss = train_one_epoch(model, train_dataloader, optimizer)

            # Validation Step
            valid_loss = evaluate_model(model, valid_dataloader)

            # Update Scheduler
            scheduler.update(valid_loss, model)
            if scheduler.early_stop():
                break

            # Update Progress Bar
            num_epochs_pbar.set_description(f"Train Loss: {train_loss.numpy()[0]:.4f}; Valid Loss: {valid_loss:.4f}; Coefficients: {model.coefficients()}")

        # Model selection cost is set to loss and penalizes a leading coefficient close to 0.
        model_cost = valid_loss + 1/abs(model.coefficients()[-1])
        print(f"Model Cost: {model_cost:.4f}, Coefficients: {scheduler.best_coefficients}")
        if model_cost < best_cost:
            best_cost = model_cost
            best_degree = degree
            best_coefficients = scheduler.best_coefficients

    # Load Best Model
    model = PolyNet()
    model_func = np.poly1d(best_coefficients[::-1])
    print(f"\nLearned Polynomial Model:\n{model_func}")
    model.load_coefficients(best_coefficients)

    input_transform = polynomial_transform(best_degree)

    # Check performance on training Set
    Ypred = model.forward(input_transform(X)).reshape((-1,1)).numpy()
    mse_train = mean_squared_error(Y, Ypred)
    r2_train = r2_score(Y, Ypred)
    print(f"Train MSE: {mse_train:.4f}, Train R^2: {r2_train:.4f}")


    X_disp = np.linspace(np.min(X_train), np.max(X_train), 100).reshape(-1, 1)
    Y_disp = model.forward(input_transform(X_disp)).reshape(-1, 1).numpy()
    plt.plot(X, Y, 'r.'), plt.plot(X_disp, Y_disp, 'b')
    plt.xlabel('x'), plt.ylabel('y')
    plt.title('Model Fit Visualization on Training Set')
    plt.legend(['Train Set', 'Learned Model'])
    plt.savefig('results/ModelFit_TrainSet.png')
    plt.show()

    # Compute MSE on Test Set
    X_test, Y_test = load_dataset(os.path.join(args.data, "data_test.csv"))
    test_dataloader = DataLoader(X_test, Y_test, batch_size, input_transform)
    mse = evaluate_model(model, test_dataloader)
    print(f"Test MSE: {mse:.4f}")

    Ypred_test = model.forward(input_transform(X_test)).reshape((-1,1)).numpy()
    mse_test = mean_squared_error(Y_test, Ypred_test)
    r2_test = r2_score(Y_test, Ypred_test)
    print(f"Test MSE: {mse_test:.4f}, Test R^2: {r2_test:.4f}")

    X_disp = np.linspace(np.min(X_test), np.max(X_test), 100).reshape(-1, 1)
    Y_disp = model.forward(input_transform(X_disp)).reshape(-1, 1).numpy()
    plt.plot(X_test, Y_test, 'go'), plt.plot(X_disp, Y_disp, 'b')
    plt.xlabel('x'), plt.ylabel('y')
    plt.title('Model Fit Visualization on Test Set')
    plt.legend(['Test Set', 'Learned Model'])
    plt.savefig('results/ModelFit_TestSet.png')
    plt.show()