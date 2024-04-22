# This code is based on PyTorch torch.nn.linear.py

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init


class CosineLinear(torch.nn.Module):
    r"""Applies a Cosine Linear transformation to the incoming data: :math:`y = xA^T / ||x|| ||A||`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = CosineLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        """
        wx / ||w|| ||x||
        """
        # with torch.no_grad():
        w_norm = torch.linalg.vector_norm(self.weight, dim=1, keepdim=True)
        x_norm = torch.linalg.vector_norm(input, dim=1, keepdim=True)
        x_w_norm = torch.matmul(x_norm, w_norm.T)
        # # print(x_norm.shape)
        # # print(w_norm.shape)
        # # print(input.shape)
        # # print(self.weight.shape)
        # # exit()
        # # return torch.matmul(input, self.weight.T) / w_norm.T
        return torch.matmul(input, self.weight.T) / x_w_norm
        """
        w and x l2 loss
        """
        # input = torch.unsqueeze(input, 1)
        # logits = torch.sum((input - self.weight)**2, dim=-1)
        # # print(logits.shape)
        # # exit()

        # return logits
        """
        just wx
        """
        # return torch.matmul(input, self.weight.T)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
