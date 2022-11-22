from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import pandas as pd



if __name__ == "__main__":
    x = Tensor.eye(3, requires_grad=True)
    y = Tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    print(x.grad)  # dz/dx
    print(y.grad)  # dz/dy

    # ... and complete like pytorch, with (x,y) data
    train_data = pd.read_csv("data_train.csv")
    print(train_data)

    # x = Tensor.eye(3, requires_grad=True)
    # out = model.forward(x)
    # loss = out.mul(y).mean()
    # optim.zero_grad()
    # loss.backward()
    # optim.step()