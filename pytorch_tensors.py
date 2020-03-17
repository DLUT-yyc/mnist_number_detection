import torch

print(torch.cuda.device_count())
device = torch.device('cuda:0')

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
# 将参数同步到显存中，如果需要
if device.type == 'cuda':
    model.cuda(device)

learning_rate = 1e-4
# 使用Adam优化代替基础梯度下降
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()

    # 用内嵌优化器更新参数
    optimizer.step()
    optimizer.zero_grad()

    print(loss)


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 保存反向传播时需要用到的变量
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_y):
        # 获得前向传播时保存的变量
        x, = ctx.saved_tensors
        grad_input = grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input


def exp3():
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    w1 = torch.randn(D_in, H, requires_grad=True,
                     device=device)
    w2 = torch.randn(H, D_out, requires_grad=True,
                     device=device)

    learning_rate = 1e-6
    for t in range(500):
        # MyReLU.apply为静态函数，
        # 会自动调用forward和backward
        y_pred = MyReLU.apply(x.mm(w1)).mm(w2)
        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()

        print(loss)


def exp2():
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    w1 = torch.randn(D_in, H, requires_grad=True,
                     device=device)
    w2 = torch.randn(H, D_out, requires_grad=True,
                     device=device)

    learning_rate = 1e-6
    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        # 此后的计算不需要建立计算图
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            # 以下划线结尾的Pytorch方法，
            # 表示修改Tensor内的值，
            # 不产生新的Tensor
            w1.grad.zero_()
            w2.grad.zero_()


def exp1():
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    w1 = torch.randn(D_in, H, device=device)
    w2 = torch.randn(H, D_out, device=device)

    learning_rate = 1e-6
    for t in range(500):
        # mm - 矩阵乘, x*w1
        h = x.mm(w1)
        # 小于0的数据被赋值为0, Max(0,h)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)
        loss = (y_pred - y).pow(2).sum()

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2



