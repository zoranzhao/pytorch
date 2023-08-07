# Owner(s): ["module: nn"]

import contextlib
from functools import partial
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.optim as optim
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, onlyCPU
from typing import List, Tuple, Union, Optional
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck
)


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import SM80OrLater, PLATFORM_SUPPORTS_FUSED_SDPA

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer


@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

isSM86or89Device = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 9)]
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5

def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol


def rand_math_tensor(shape: Tuple[Union[int, List[int]]], device: str, dtype: torch.dtype,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)

def init_tensor(
        tensor_list,
        **kwargs
) -> torch.Tensor:
    return torch.Tensor(tensor_list).to(**kwargs)


def test_comp_nocomp(
        function,
        *inputs,
        **kwargs
):
    c_function=torch.compile(function)

    f_res = function(*inputs)
    cf_res = c_function(*inputs)

    torch.testing.assert_close(f_res, cf_res, **kwargs)


# The test functions are used by several tests
def test_mm(a, b):
    return torch.mm(a, b)

def test_addmm(add, b, c):
    return torch.addmm(add, b, c)

def test_bmm(a, b):
    return torch.bmm(a, b)

def test_baddbmm(add, b, c, alpha, beta):
    return torch.baddbmm(add, b, c, alpha=alpha, beta=beta)

# The shapes we test on
ts_list = [(1, 32, 32,1,), (1, 10, 10, 1), (1, 3, 3, 1), (32, 1, 1, 32), (3, 1, 1, 3), (4, 1, 1, 9), (9, 1, 1, 4)]

class TestDecomp(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_simple_mm(self, device, dtype):

        fudge = 3
        rtol=default_rtol[dtype] * fudge
        atol=default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1=rand_math_tensor((a1_0, a1_1), dtype=dtype, device=device)
            t2=rand_math_tensor((a2_0, a2_1), dtype=dtype, device=device)
            tadd=rand_math_tensor((a1_0, a2_1), dtype=dtype, device=device)

            test_comp_nocomp(test_mm, t1, t2, rtol=rtol, atol=atol)
            test_comp_nocomp(test_addmm, tadd, t1, t2, rtol=rtol, atol=atol)


    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("bs", [1, 2, 4, 10])
    def test_batched_mm(self, device, dtype, bs):

        fudge = 3
        rtol=default_rtol[dtype] * fudge
        atol=default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1=rand_math_tensor((bs, a1_0, a1_1), dtype=dtype, device=device)
            t2=rand_math_tensor((bs, a2_0, a2_1), dtype=dtype, device=device)
            tadd=rand_math_tensor((bs, a1_0, a2_1), dtype=dtype, device=device)

            test_comp_nocomp(test_bmm, t1, t2, rtol=rtol, atol=atol)

            for alpha in (0, 1, -1, 0.5, -0.5):
                for beta in (0, 1, -1, 0.5, -0.5):
                    test_comp_nocomp(test_baddbmm, tadd, t1, t2, alpha, beta, rtol=rtol, atol=atol)


    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    def test_some(self, device, dtype):
        test_comp_nocomp(
            test_mm,
            init_tensor([[1],[2],[3],[4]], dtype=dtype, device=device),
            init_tensor([[1, 2 , 3, 4]], dtype=dtype, device=device),
        )
        test_comp_nocomp(
            test_mm,
            init_tensor([[1, 2 , 3, 4]], dtype=dtype, device=device),
            init_tensor([[1],[2],[3],[4]], dtype=dtype, device=device),
        )

    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    @parametrize("bs", [1, 2, 4, 10])
    def test_some_batched(self, device, dtype, bs):
        test_comp_nocomp(
            test_bmm,
            init_tensor([[[1],[2],[3],[4]]] * bs , dtype=dtype, device=device),
            init_tensor([[[1, 2 , 3, 4]]] * bs, dtype=dtype, device=device),
        )
        test_comp_nocomp(
            test_bmm,
            init_tensor([[[1, 2 , 3, 4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1],[2],[3],[4]]] * bs, dtype=dtype, device=device),
        )


device_types = ("cpu", "cuda")
instantiate_device_type_tests(TestDecomp, globals(), only_for=device_types)

if __name__ == '__main__':
    run_tests()
