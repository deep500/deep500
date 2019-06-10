from typing import Dict

from torch import nn
import torch.nn.functional as F
import torch
from torch import autograd
from torch.nn import Module

from deep500.frameworks.pytorch.pytorch_network import PyTorchNetwork
from deep500.utils.onnx_interop.generated_operators import *
from deep500.utils.onnx_interop.losses import *
from deep500.utils.onnx_interop.onnx_objects import (OnnxTensor,
    OnnxFloatTensor, OnnxDoubleTensor, OnnxValueInfo)
from deep500.utils.onnx_interop.onnx_base_visitor import OnnxBaseVisitor, EmptyOnnxBaseVisitor


def _is_param(tensor: OnnxTensor):
    """ A helper function to deal with an ONNX limitation - Tensors cannot be
        designated as parameters or fixed. Returns True if the tensor should
        be treated as a parameter (i.e., require gradients), or False
        otherwise. """
    if not isinstance(tensor, (OnnxFloatTensor, OnnxDoubleTensor)):
        return False
    if 'running_mean' in tensor.name or 'running_var' in tensor.name:
        return False

    return True


class PyTorchMetaVisitor(EmptyOnnxBaseVisitor):

    def visit_initializer(self, each_initializer: OnnxTensor, network: PyTorchNetwork):
        network.feed_tensor(each_initializer.name, each_initializer.get_data(),
                            is_param=_is_param(each_initializer))

    def visit_net_output(self, output: OnnxValueInfo, network: PyTorchNetwork):
        network.outputs[output.name] = output.name

    def visit_label_cross_entropy(self, label_cross_entropy: LabelCrossEntropy, network: PyTorchNetwork):
        network.outputs[label_cross_entropy.o_output] = label_cross_entropy.o_output


class PyTorchVisitor(OnnxBaseVisitor):

    def __init__(self):
        self.initializer_visited = False
        # globals obs
        self.global_ops = {} # type: Dict[str, Module]
        # optimizer
        self.optimizer = None

    def visit_constant(self, op: Constant, network: PyTorchNetwork):
        network.feed_tensor(op.o_output, op.value.get_value())

    def visit_add(self, op: Add, network: PyTorchNetwork):
        A = network.fetch_tensor_internal(op.i_A)
        B = network.fetch_tensor_internal(op.i_B)
        # if op.broadcast is not None and op.broadcast.get_value() != 0:
        #     B_new_shape = self.get_broadcast_shape(A, B, op.axis.get_value())
        #     B = B.view(B_new_shape)
        network.feed_tensor(op.o_C, A + B)

    def visit_abs(self, op: Abs, network: PyTorchNetwork):
        X = network.fetch_tensors_internal([op.i_X])[0]
        network.feed_tensor(op.o_Y, torch.abs(X))

    def visit_sub(self, op: Sub, network: PyTorchNetwork):
        A, B = network.fetch_tensors_internal([op.i_A, op.i_B])
        A = torch.mul(A, (-1))
        network.feed_tensor(op.o_C, torch.add(A, B))

    def visit_pow(self, op: Pow, network: PyTorchNetwork):
        X, Y = network.fetch_tensors_internal([op.i_X, op.i_Y])
        network.feed_tensor(op.o_Z, torch.pow(X, Y))

    def visit_reshape(self, op: Reshape, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_data)
        S = list(network.fetch_tensor(op.i_shape))
        network.feed_tensor(op.o_reshaped, X.view(S))

    def visit_conv(self, op: Conv, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        W = network.fetch_tensor_internal(op.i_W)
        B = None if op.i_B is None else network.fetch_tensor_internal(op.i_B)
        args = self.get_conv_base(op)
        result = F.conv2d(X, W, B, **args)
        network.feed_tensor(op.o_Y, result)

    def visit_pad(self, op, network: PyTorchNetwork):
        data = network.fetch_tensor_internal(op.i_data)
        result = F.pad(data, op.pads.value, mode=op.mode.value, value=op.value.value)
        network.feed_tensor(op.o_output, result)

    def visit_shape(self, op, network: PyTorchNetwork):
        data = network.fetch_tensor_internal(op.i_data)
        network.feed_tensor(op.o_shape, torch.tensor(data.shape))

    def visit_gather(self, op, network: PyTorchNetwork):
        input = network.fetch_tensor_internal(op.i_data)
        indices = network.fetch_tensor_internal(op.i_indices)
        results = torch.gather(
            input,
            op.axis.value,
            indices)
        network.feed_tensor(op.o_output, results)

    def visit_unsqueeze(self, op, network: PyTorchNetwork):
        data = network.fetch_tensor_internal(op.i_data)
        result = data
        for axis in op.axes.value:
            result = torch.unsqueeze(result, axis)
        network.feed_tensor(op.o_expanded, result)

    def visit_lrn(self, op, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        result = F.local_response_norm(
            X,
            op.size.value,
            op.alpha.value,
            op.beta.value,
            op.bias.value)
        network.feed_tensor(op.o_Y, result)

    def visit_split(self, op, network):
        input = network.fetch_tensor_internal(op.i_input)
        results = torch.split(input, op.split.value, dim=op.axis.value)
        for res,out in zip(results, op.output):
            network.feed_tensor(out, res)

    def visit_concat(self, op, network):
        in_tensors = [network.fetch_tensor_internal(i) for i in op.input]
        out = torch.cat(in_tensors, dim=op.axis.value)
        network.feed_tensor(op.output[0], out)

    def visit_maxpool(self, op: MaxPool, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        args = {}
        if hasattr(op, 'kernel_shape') and op.kernel_shape:
            args['kernel_size'] = op.kernel_shape.get_value()
        if hasattr(op, 'strides') and op.strides:
            args['stride'] = op.strides.get_value()[0]
        if hasattr(op, 'pads') and op.pads:
            args['padding'] = op.pads.get_value()[0]
        result = F.max_pool2d(X, **args)
        network.feed_tensor(op.o_Y, result)

    def visit_averagepool(self, op: AveragePool, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        args = {}
        if hasattr(op, 'kernel_shape') and op.kernel_shape:
            args['kernel_size'] = op.kernel_shape.get_value()
        if hasattr(op, 'strides') and op.strides:
            args['stride'] = op.strides.get_value()[0]
        if hasattr(op, 'pads') and op.pads:
            args['padding'] = op.pads.get_value()[0]
        result = F.avg_pool2d(X, **args)
        network.feed_tensor(op.o_Y, result)

    def visit_gemm(self, op: Gemm, network: PyTorchNetwork):
        A, B, C = network.fetch_tensors_internal([op.i_A, op.i_B, op.i_C])
        trans_a = 0 if op.transA is None else op.transA.get_value()
        trans_b = 0 if op.transB is None else op.transB.get_value()

        if op.alpha:
            A = torch.mul(A, op.alpha.get_value())
        if op.beta:
            C = torch.mul(C, op.beta.get_value())

        if trans_a:
            A = A.transpose(len(A.shape) - 1, len(A.shape) - 2)
        if trans_b:
            B = B.transpose(len(B.shape) - 1, len(B.shape) - 2)

        A = torch.matmul(A, B)
        result = torch.add(A, C)
        network.feed_tensor(op.o_Y, result)

    def visit_label_cross_entropy(self, op: LabelCrossEntropy, network: PyTorchNetwork):
        X, label = network.fetch_tensors_internal([op.i_X, op.i_target])
        network.feed_tensor(op.o_output, F.cross_entropy(X, label.long()))

    def visit_relu(self, op: Relu, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        network.feed_tensor(op.o_Y, F.relu(X))

    def visit_softmax(self, op: Softmax, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_input)
        network.feed_tensor(op.o_output, F.relu(X))

    def visit_batchnormalization(self, op: BatchNormalization, network: PyTorchNetwork):
        X = network.fetch_tensor_internal(op.i_X)
        mean = network.fetch_tensor_internal(op.i_mean)
        var = network.fetch_tensor_internal(op.i_var)
        weight = network.fetch_tensor_internal(op.i_scale)
        B = network.fetch_tensor_internal(op.i_B)
        epsilon = op.epsilon.get_value() if op.epsilon else 1e-5
        momentum = op.momentum.get_value() if op.momentum else 0.1

        network.feed_tensor(op.o_Y, F.batch_norm(X, mean, var, weight, B, momentum=momentum, eps=epsilon))

    def get_conv_base(self, op):
        args = {}
        if hasattr(op, 'strides') and op.strides:
            args['stride'] = op.strides.get_value()[0]
        if hasattr(op, 'padding') and op.padding:
            args['padding'] = op.padding.get_value()[0]
        if hasattr(op, 'pads') and op.pads:
            args['padding'] = op.pads.get_value()[0]
        if hasattr(op, 'dilations') and op.dilations:
            args['dilation'] = op.dilations.get_value()[0]
        if hasattr(op, 'group') and op.group:
            args['groups'] = op.group.get_value()

        return args

    def get_broadcast_shape(self, A, B, axis):
        A_tensor = A.data if isinstance(A, autograd.Variable) else A
        B_tensor = B.data if isinstance(B, autograd.Variable) else B
        A_shape = A_tensor.size()
        B_shape = B_tensor.size()

        if len(B_shape) > 1:
            # TODO(HBS): currently only supporting broadcast if shape of B is [1]. add more possibilities
            raise NotImplementedError

        n = len(A_shape)
        B_new_shape = [1] * n
        B_new_shape[axis] = B_shape[0]

        return B_new_shape
