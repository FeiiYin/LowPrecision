import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("points.png")


def origin_softmax(input):

    # type : (Tensor) -> Tensor
    output = torch.IntTensor(input.size())

    ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            output[now_dim][now_len] = input[now_dim][now_len] * ratio

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        now_dim_max = torch.max(output[now_dim])
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            output[now_dim][now_len] -= (now_dim_max + 2)     # this value could change
            # print(input[now_dim][now_len])

    output = output.float()
    output = output.mul(1.0/ratio)

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            if output[now_dim, now_len] < -20:
                output[now_dim, now_len] = -20

    return output


def simulation_softmax(input):

    # type : (Tensor) -> Tensor
    output = torch.IntTensor(input.size())

    ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            output[now_dim][now_len] = input[now_dim][now_len] * ratio

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        now_dim_max = torch.max(output[now_dim])
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            output[now_dim][now_len] -= (now_dim_max + 2)     # this value could change
            # print(input[now_dim][now_len])

    output = output.float()
    output = output.mul(1.0/ratio)

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            if output[now_dim][now_len] < -20:
                output[now_dim][now_len] = -20

    return output


def simulation_softmax_init(input):
    # type : (Tensor) -> Tensor
    output = torch.IntTensor(input.size())

    ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            output[now_dim, now_len] = input[now_dim, now_len] * ratio

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        now_dim_max = torch.max(output[now_dim])
        now_dim_min = torch.min(output[now_dim])
        now_hash = ratio * 1.0 / (now_dim_max - now_dim_min)
        for now_len in range(0, len):
            output[now_dim, now_len] = (output[now_dim, now_len] - now_dim_min) * now_hash
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            output[now_dim, now_len] -= (ratio + 2)  # this value could change
            # print(input[now_dim][now_len])

    output = output.float()
    output = output.mul(1.0 / ratio)

    # for now_dim in range(0, dim):
    #     for now_len in range(0, len):
    #         if output[now_dim, now_len] < -20:
    #             output[now_dim, now_len] = -20

    # return torch.from_numpy(output.cuda().data.cpu().numpy())
    return output


x = torch.FloatTensor([[0.1, 0.2], [0.2, 0.21]])
y = F.log_softmax(x, dim=-1)
z = simulation_softmax(x)
q = simulation_softmax_init(x)

print(x)
print(y)
print(z)
print(q)

# def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
#     # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
#     r"""Applies a softmax followed by a logarithm.
#
#     While mathematically equivalent to log(softmax(x)), doing these two
#     operations separately is slower, and numerically unstable. This function
#     uses an alternative formulation to compute the output and gradient correctly.
#
#     See :class:`~torch.nn.LogSoftmax` for more details.
#
#     Arguments:
#         input (Tensor): input
#         dim (int): A dimension along which log_softmax will be computed.
#         dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
#         If specified, the input tensor is casted to :attr:`dtype` before the operation
#         is performed. This is useful for preventing data type overflows. Default: None.
#     """
#     if dim is None:
#         dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
#     else:
#         dim = torch.jit._unwrap_optional(dim)
#     if dtype is None:
#         ret = input.log_softmax(dim)
#     else:
#         _dtype = torch.jit._unwrap_optional(dtype)
#         ret = input.log_softmax(dim, dtype=_dtype)
#     return ret






#
# class LogSoftmax(Module):
#     r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
#     input Tensor. The LogSoftmax formulation can be simplified as:
#
#     .. math::
#         \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
#
#     Shape:
#         - Input: any shape
#         - Output: same as input
#
#     Arguments:
#         dim (int): A dimension along which Softmax will be computed (so every slice
#             along dim will sum to 1).
#
#     Returns:
#         a Tensor of the same dimension and shape as the input with
#         values in the range [-inf, 0)
#
#     Examples::
#
#         >>> m = nn.LogSoftmax()
#         >>> input = torch.randn(2, 3)
#         >>> output = m(input)
#     """
#     __constants__ = ['dim']
#
#     def __init__(self, dim=None):
#         super(LogSoftmax, self).__init__()
#         self.dim = dim
#
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         if not hasattr(self, 'dim'):
#             self.dim = None
#
#     @weak_script_method
#     def forward(self, input):
#         return F.log_softmax(input, self.dim, _stacklevel=5)
#
#
# Tensor.softmax