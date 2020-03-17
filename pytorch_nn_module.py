import torch
from torch.utils.data import TensorDataset, DataLoader

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size=8)
model = TwoLayerNet(D_in, H, D_out)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for epoch in range(20):
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)


# 将整个模型定义在一个新的模块中
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # 初始化2个子模块，模块可以包含其他模块
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        # 利用子模块定义前向传播，
        # autograd会负责反向传播
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class ParallelBlock(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ParallelBlock, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.linear2 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        return (h1 * h2).clamp(min=0)


def exp2():
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = torch.nn.Sequential(
        ParallelBlock(D_in, H),
        ParallelBlock(H, H),
        torch.nn.Linear(H, D_out)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)


def exp1():
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = TwoLayerNet(D_in, H, D_out)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for t in range(500):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)
