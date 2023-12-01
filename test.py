from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from examples.stable_diffusion import CrossAttention
import numpy as np

Device.DEFAULT = "WEBGPU"

# # create some tensors
# batch_sz = 8
# query_len = 1000
# query_dim = 64
# key_len = 2000
# key_dim = 128
# n_heads = 10
# head_dim = 16

# query = Tensor.randn(batch_sz, query_len, query_dim)
# context = Tensor.randn(batch_sz, key_len, key_dim)

# cross_attn = CrossAttention(query_dim, key_dim, n_heads, head_dim)

# # forward pass
# out = cross_attn(query, context)
# print(out.numpy().mean())

# A = Tensor([[1, 2, 3], [4, 5, 6]])
# B = Tensor([[1, 2], [3, 4], [5, 6]])
# C = A @ B
# print(C.numpy().mean())

A = Tensor.randn(2000, 3000)
B = Tensor.randn(3000, 4000)
C = A @ B
print(C.numpy().mean())