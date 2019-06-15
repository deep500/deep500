from .custom_operators.tf import custom_op, desc_from_tensor, custom_op_from_native

from .tf_network import TensorflowNetwork
from .tf_visitor_impl import TensorflowVisitor
from .op_validation import test_nativeop_forward, test_nativeop_gradient

from .tf_graph_executor import *

from .tf_optimizers import *
try:
    from .tf_distributed_optimizer import HorovodDistributedOptimizer
except ImportError:
    pass

import deep500 as d5


def from_onnx(onnx_file: str, device: d5.DeviceType = d5.GPUDevice()) -> TensorflowGraphExecutor:
    model = d5.parser.load_and_parse_model(onnx_file)
    return from_model(model, device)


def from_model(model: d5.ops.OnnxModel, device: d5.DeviceType = d5.GPUDevice()) -> TensorflowGraphExecutor:
    return TensorflowGraphExecutor(model, device)
