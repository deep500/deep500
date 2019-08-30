# Initialize CUDA first (due to an initialization issue)
try:
    import torch.cuda
    torch.cuda.init()
except (RuntimeError, AssertionError):
    pass

from .custom_operators.pytorch import custom_op, desc_from_tensor, custom_op_from_native
from .op_validation import test_nativeop_forward, test_nativeop_gradient
import deep500 as d5

from .pytorch_network import PyTorchNetwork, PyTorchNativeNetwork
from .pytorch_visitor import PyTorchVisitor
from .pytorch_graph_executor import (PyTorchGraphExecutor,
                                     PyTorchNativeGraphExecutor)
from .pytorch_optimizers import *


def from_onnx(onnx_file: str, device: d5.DeviceType = d5.GPUDevice()) -> PyTorchGraphExecutor:
    model = d5.parser.load_and_parse_model(onnx_file)
    return from_model(model, device)


def from_model(model: d5.ops.OnnxModel, device: d5.DeviceType = d5.GPUDevice()) -> PyTorchGraphExecutor:
    graph_executor = PyTorchGraphExecutor(model, device)
    return graph_executor
