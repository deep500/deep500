from typing import Dict, Union

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

#
class PyTorchMetaVisitor(EmptyOnnxBaseVisitor):

    def visit_initializer(self, each_initializer: OnnxTensor, network: PyTorchNetwork):
        network.feed_tensor(each_initializer.name, each_initializer.get_data(),
                            is_param=_is_param(each_initializer))

    def visit_net_output(self, output: OnnxValueInfo, network: PyTorchNetwork):
        network.outputs[output.name] = output.name

    def visit_softmax_cross_entropy(self, label_cross_entropy: SoftmaxCrossEntropy, network: PyTorchNetwork):
        network.outputs[label_cross_entropy.o_output] = label_cross_entropy.o_output


class GraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._params = {}
        self._outputs = set()
        self._nomove = {}
        self._compute = []

    def forward(self):
        for func, out, args in self._compute:
            output = func(*(self._params.get(arg, arg)
                                       for arg in args))
            if isinstance(output, list):
                for o, oname in zip(output, out):
                    self._params[oname] = o
            else:
                self._params[out] = output


        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for pname, p in self._params.items():
            if pname not in self._nomove:
                self._params[pname] = p.to(*args, **kwargs)
        return self

# TODO: Initialize conv and batchnorm (initializers!)
class PyTorchVisitor(OnnxBaseVisitor):

    def __init__(self):
        self.model = GraphModule()
        self._tensors = {}

    def _get_shape(self, name):
        return self._tensors[name].type.shape.shape

    def visit_net_input(self, input: OnnxValueInfo, network: PyTorchNetwork):
        self._tensors[input.name] = input

    def visit_net_output(self, output: OnnxValueInfo, network: PyTorchNetwork):
        self._tensors[output.name] = output
        self.model._outputs.add(output.name)

    def visit_initializer(self, each_initializer: OnnxTensor, network: PyTorchNetwork):
        self._add_param(each_initializer.name,
                        torch.tensor(each_initializer.get_data()),
                        trainable=False)#_is_param(each_initializer))

    def _add_param(self, name, value, trainable=False, nomove=False):
        self.model._params[name] = value
        if nomove:
            self.model._nomove[name] = True
        if trainable:
            self.model.register_parameter(name, value)

    def _add_computation(self, module, output, arguments):
        self.model._compute.append((module, output, arguments))
        if isinstance(module, torch.nn.Module):
            self.model.add_module(output, module)

    def general_visit(self, op: Operation, result: Union[nn.Module, torch.Tensor]):
        # Set the result as a sub-module/sub-buffer
        self._add_computation(op.output[0], result, tuple())

    def visit_constant(self, op: Constant, network: PyTorchNetwork):
        self._add_param(op.output[0], torch.tensor(op.value.get_value()), nomove=True)

    def visit_add(self, op: Add, network: PyTorchNetwork):
        self._add_computation(lambda a, b: a + b, op.o_C, (op.i_A, op.i_B))

    def visit_abs(self, op: Abs, network: PyTorchNetwork):
        self._add_computation(torch.abs, op.o_Y, (op.i_X,))

    def visit_sub(self, op: Sub, network: PyTorchNetwork):
        self._add_computation(lambda a, b: a - b, op.o_C, (op.i_A, op.i_B))

    def visit_pow(self, op: Pow, network: PyTorchNetwork):
        self._add_computation(torch.pow, op.o_Z, (op.i_X, op.i_Y))

    def visit_reshape(self, op: Reshape, network: PyTorchNetwork):
        self._add_computation(
            lambda data, shape: torch.reshape(data, shape.tolist()),
            op.o_reshaped, (op.i_data, op.i_shape))

    def visit_conv(self, op: Conv, network: PyTorchNetwork):
        kwargs = self.get_conv_base(op)

        # Assuming NCHW data layout due to ONNX
        conv_shape = self._get_shape(op.i_W)

        # Set module parameters
        mod = torch.nn.Conv2d(conv_shape[1], conv_shape[0], op.kernel_shape.value,
                              **kwargs)
        # with torch.no_grad():
        #     mod.weight[:] = network.fetch_internal_tensor(op.i_W)
        #     mod.bias[:] = network.fetch_internal_tensor(op.i_B)
        self._add_computation(mod, op.o_Y, (op.i_X,))

    def visit_pad(self, op, network: PyTorchNetwork):
        self._add_computation(lambda x: F.pad(x, op.pads.value,
                                              mode=op.mode.value,
                                              value=op.value.value),
                              op.o_output, (op.i_data,))

    def visit_shape(self, op, network: PyTorchNetwork):
        self._add_computation(lambda x: torch.tensor(x.shape),
                              op.o_shape, (op.i_data,))

    def visit_gather(self, op, network: PyTorchNetwork):
        self._add_computation(lambda x, i: torch.gather(x, op.axis.value, i),
                              op.o_output, (op.i_data, op.i_indices))

    def visit_unsqueeze(self, op, network: PyTorchNetwork):
        def unsqueeze(x):
            result = x
            for axis in op.axes.value:
                result = torch.unsqueeze(result, axis)
            return result
        self._add_computation(unsqueeze, op.o_expanded,
                              (op.i_data,))

    def visit_lrn(self, op, network: PyTorchNetwork):
        mod = torch.nn.LocalResponseNorm(op.size.value, op.alpha.value,
                                         op.beta.value, op.k.value)
        self._add_computation(mod, op.o_Y, (op.i_X,))

    def visit_split(self, op, network):
        self._add_computation(
            lambda x: torch.split(x, op.split.value, dim=op.axis.value),
            op.output, (op.i_input,))

    def visit_concat(self, op, network):
        self._add_computation(
            lambda *args: torch.cat(args, dim=op.axis.value),
            op.output[0], op.input)

    def _visit_pool(self, op, op_name: str):
        kernel_shape = op.kernel_shape.get_value()

        # Dynamically obtain module type
        modtyp = getattr(torch.nn, '%s%dd' % (op_name, len(kernel_shape)), None)
        if modtyp is None:
            raise RuntimeError('Unsupported dimensions for %s (%d)' %
                               (op_name, len(kernel_shape)))

        mod = modtyp(kernel_shape, op.strides.value[0], op.pads.value[0])

        self._add_computation(mod, op.o_Y, (op.i_X,))

    def visit_maxpool(self, op: MaxPool, network: PyTorchNetwork):
        self._visit_pool(op, 'MaxPool')

    def visit_averagepool(self, op: AveragePool, network: PyTorchNetwork):
        self._visit_pool(op, 'AvgPool')

    def visit_gemm(self, op: Gemm, network: PyTorchNetwork):
        alpha = op.alpha.value if op.alpha is not None else 1.0
        beta = op.beta.value if op.beta is not None else 0.0
        trans_a = 0 if op.transA is None else op.transA.value
        trans_b = 0 if op.transB is None else op.transB.value

        def gemm(A, B, C):
            if alpha != 1.0 and alpha != 0.0:
                A = torch.mul(A, alpha)
            if beta != 1.0 and beta != 0.0 and C is not None:
                C = torch.mul(C, beta)
            if trans_a:
                A = A.transpose(len(A.shape) - 1, len(A.shape) - 2)
            if trans_b:
                B = B.transpose(len(B.shape) - 1, len(B.shape) - 2)

            result = torch.matmul(A, B)
            if beta != 0.0 and C is not None:
                result += C
            return result

        self._add_computation(gemm, op.o_Y, (op.i_A, op.i_B, op.i_C))

    def visit_softmax_cross_entropy(self, op: SoftmaxCrossEntropy, network: PyTorchNetwork):
        mod = torch.nn.CrossEntropyLoss()
        self._add_computation(mod, op.o_output, (op.i_X, op.i_target))
        self.model._outputs.add(op.o_output)

    def visit_relu(self, op: Relu, network: PyTorchNetwork):
        mod = torch.nn.ReLU()
        self._add_computation(mod, op.o_Y, (op.i_X,))

    def visit_softmax(self, op: Softmax, network: PyTorchNetwork):
        mod = torch.nn.Softmax()
        self._add_computation(mod, op.o_Y, (op.i_X,))

    def visit_batchnormalization(self, op: BatchNormalization, network: PyTorchNetwork):
        epsilon = op.epsilon.get_value() if op.epsilon else 1e-5
        momentum = op.momentum.get_value() if op.momentum else 0.9
        #mean = network.fetch_internal_tensor(op.i_mean)
        #var = network.fetch_internal_tensor(op.i_var)
        #scale = network.fetch_internal_tensor(op.i_scale)
        #bias = network.fetch_internal_tensor(op.i_B)

        # PyTorch uses the complement momentum
        momentum = 1. - momentum

        bn = torch.nn.BatchNorm2d(self._get_shape(op.i_scale)[0], eps=epsilon,
                                  momentum=momentum)
        #with torch.no_grad():
        #    bn.weight[:] = scale
        #    bn.bias[:] = bias
        #    bn.running_mean[:] = mean
        #    bn.running_var[:] = var

        self._add_computation(bn, op.o_Y, (op.i_X,))

    def visit_dropout(self, op: Dropout, network: PyTorchNetwork):
        mod = torch.nn.Dropout(p=op.ratio.value)
        self._add_computation(mod, op.o_output, (op.i_data,))

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
