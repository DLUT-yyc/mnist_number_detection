import numpy as np
np.random.seed(0)

# N - 样本个数 D - 样本维度
N, D = 3, 4

# 从标准正态分布中生成 NxD 样本
x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
# axis=0 - 按行求和, axis=1 - 按列求和,
# axis=None - 所有元素求和
c = np.sum(b, axis=None)

grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
print('-'*64)
print('[INFO: NUMPY NN] y==grad_x?\n{}'.format(y == grad_x))
print('-'*64)
# Numpy优点是API清晰, 代码简单明了, 缺点是需要手动计算梯度，不能运行在GPU上

# Pytorch实现相同功能代码如下
import torch

device = 'cuda:0'
x = torch.randn(N, D, requires_grad=True,
                device=device)
y = torch.randn(N, D, device=device)
z = torch.randn(N, D, device=device)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()
print('-'*64)
print('[INFO: PYTORCH NN] y==grad_x?\n{}'.format(y == x.grad))
print('-'*64)
