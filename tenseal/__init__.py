"""TenSEAL is a library for doing homomorphic encryption operation on tensors.
"""

try:
    import _tenseal_cpp as _ts_cpp
except ImportError:
    import tenseal._tenseal_cpp as _ts_cpp
from tenseal.tensors import CKKSTensor, CKKSVector, BFVVector, BFVTensor, PlainTensor

from tenseal.enc_context import Context, SCHEME_TYPE, ENCRYPTION_TYPE
from tenseal.version import __version__
import numpy as np
import torch

def im2col_encoding(
    context: Context, tensor, kernel_n_rows: int, kernel_n_cols: int, stride: int, size: int = None
) -> CKKSVector:
    """Encoding an image into a CKKSVector. This serves for doing efficient Conv2d evaluation.

    Args:
        context: a Context object, holding the encryption parameters and keys.
        tensor: tensor-like object.
        kernel_n_rows: number of rows in the kernel that will be used for conv2d.
        kernel_n_cols: number of columns in the kernel that will be used for conv2d.
        stride: stride that will be used for conv2d.

    Returns:
        Encrypted image into a CKKSVector.
    """
    if not isinstance(context, Context):
        raise TypeError("context must be of type tenseal.Context")
    if not isinstance(tensor, PlainTensor):

        tensor = plain_tensor(tensor)

        if len(tensor.shape) != 2:
            raise ValueError("tensor must be a matrix")
        matrix = tensor.tolist()
    
    if size is not None:
        ckks_vec, windows_nb = _ts_cpp.new_im2col_encoding(
            context.data, matrix, kernel_n_cols, kernel_n_rows, stride, context.slot()
        )
    else:
        ckks_vec, windows_nb = _ts_cpp.im2col_encoding(
            context.data, matrix, kernel_n_cols, kernel_n_rows, stride
        )

    return CKKSVector._wrap(ckks_vec), windows_nb


def enc_matmul_encoding(context: Context, tensor) -> CKKSVector:
    """Encode a matrix into a CKKSVector for later matrix(encrypted)-vector(plain) multiplication.

    Args:
        context: a Context object, holding the encryption parameters and keys.
        tensor: tensor-like object of shape 2 (matrix).

    Returns:
        Encrypted matrix into a CKKSVector.
    """
    if not isinstance(context, Context):
        raise TypeError("context must be of type tenseal.Context")
    if not isinstance(tensor, PlainTensor):
        tensor = plain_tensor(tensor)
        if len(tensor.shape) != 2:
            raise ValueError("tensor must be a matrix")
        matrix = tensor.tolist()
    return CKKSVector._wrap(_ts_cpp.enc_matmul_encoding(context.data, matrix))


def context(*args, **kwargs) -> Context:
    """Constructor function for tenseal.Context"""
    return Context(*args, **kwargs)


def context_from(data: bytes, n_threads: int = None) -> Context:
    """Load a Context from a protocol buffer.
    n_threads set the concurrency for the context if parallel computation is requested."""
    return Context.load(data, n_threads)


def plain_tensor(*args, **kwargs) -> PlainTensor:
    """Constructor function for tenseal.PlainTensor"""
    return PlainTensor(*args, **kwargs)


def plain_tensor_from(data: bytes, dtype: str = "float") -> PlainTensor:
    """Load a PlainTensor from a buffer."""
    return PlainTensor.load(data, dtype)


def bfv_vector(*args, **kwargs) -> BFVVector:
    """Constructor function for tenseal.BFVVector"""
    return BFVVector(*args, **kwargs)


def bfv_vector_from(context: Context, data: bytes) -> BFVVector:
    """Load a BFVVector from a protocol buffer.
    Requires the context to be linked with."""
    return BFVVector.load(context, data)


def lazy_bfv_vector_from(data: bytes) -> BFVVector:
    """Load a BFVVector from a protocol buffer."""
    return BFVVector.lazy_load(data)


def ckks_vector(*args, **kwargs) -> CKKSVector:
    """Constructor function for tenseal.CKKSVector"""
    return CKKSVector(*args, **kwargs)


def ckks_vector_from(context: Context, data: bytes) -> CKKSVector:
    """Load a CKKSVector from a protocol buffer.
    Requires the context to be linked with."""
    return CKKSVector.load(context, data)


def lazy_ckks_vector_from(data: bytes) -> CKKSVector:
    """Load a CKKSVector from a protocol buffer."""
    return CKKSVector.lazy_load(data)


def ckks_tensor(*args, **kwargs) -> CKKSTensor:
    """Constructor function for tenseal.CKKSTensor"""
    return CKKSTensor(*args, **kwargs)


def ckks_tensor_from(context: Context, data: bytes) -> CKKSTensor:
    """Load a CKKSTensor from a protocol buffer.
    Requires the context to be linked with."""
    return CKKSTensor.load(context, data)


def lazy_ckks_tensor_from(data: bytes) -> CKKSTensor:
    """Load a CKKSTensor from a protocol buffer"""
    return CKKSTensor.lazy_load(data)


def bfv_tensor(*args, **kwargs) -> BFVTensor:
    """Constructor function for tenseal.BFVTensor"""
    return BFVTensor(*args, **kwargs)


def bfv_tensor_from(context: Context, data: bytes) -> BFVTensor:
    """Load a BFVTensor from a protocol buffer.
    Requires the context to be linked with."""
    return BFVTensor.load(context, data)


def lazy_bfv_tensor_from(data: bytes) -> BFVTensor:
    """Load a BFVTensor from a protocol buffer"""
    return BFVTensor.lazy_load(data)


__all__ = [
    "bfv_vector",
    "bfv_vector_from",
    "lazy_bfv_vector_from",
    "ckks_vector",
    "ckks_vector_from",
    "lazy_ckks_vector_from",
    "ckks_tensor",
    "ckks_tensor_from",
    "lazy_ckks_tensor_from",
    "bfv_tensor",
    "bfv_tensor_from",
    "lazy_bfv_tensor_from",
    "context",
    "context_from",
    "im2col_encoding",
    "plain_tensor",
    "plain_tensor_from",
    "ENCRYPTION_TYPE",
    "SCHEME_TYPE",
    "__version__",
]
