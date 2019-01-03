from caffe2.proto import caffe2_pb2
from caffe2.python import core

import deep500 as d5

from .caffe2_graph_executor import Caffe2GraphExecutor

from .op_validation import test_nativeop_forward


def _get_device(device_option: d5.DeviceType) -> core.DeviceOption:
    device = core.DeviceOption(caffe2_pb2.CPU)
    if device_option.is_gpu():
        device = core.DeviceOption(caffe2_pb2.CUDA)
    return device


def from_onnx(onnx_file: str, device: d5.DeviceType = d5.GPUDevice()) -> Caffe2GraphExecutor:
    model = d5.parser.load_and_parse_model(onnx_file)
    return from_model(model, device)


def from_model(model: d5.ops.OnnxModel, device: d5.DeviceType = d5.GPUDevice()) -> Caffe2GraphExecutor:
    return Caffe2GraphExecutor(model, _get_device(device))