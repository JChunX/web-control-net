from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True, device='webgpu')
y = Tensor([[2.0,0,-2.0]], requires_grad=True, device='webgpu')
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy