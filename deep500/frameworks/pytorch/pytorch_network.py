import numpy as np
import torch
from torch import autograd
from typing import List

import deep500 as d5


class PyTorchNetwork(d5.Network):
    def __init__(self, device: d5.DeviceType):
        super(PyTorchNetwork, self).__init__()
        self.variables = {}
        self.inputs = {}
        self.outputs = set()
        self.optimizer_created = False
        self.grad_params = {}
        self.cuda = device.is_gpu()
        if self.cuda:
            torch.cuda.set_device(device.num)

    @property
    def parameters(self):
        return [self.variables[k] for k in self.grad_params.keys()]

    def get_params(self):
        return list(self.grad_params.keys())

    def get_input_nodes(self) -> List[str]:
        return list(self.inputs.keys())

    def get_output_nodes(self) -> List[str]:
        return list(self.outputs.keys())
        
    def _feed_input(self, input):
        for key, value in input.items():
            self.feed_tensor(key, value)

    def _save_gradients(self):
        for (param, grad_param) in self.grad_params.items():
            self.variables[grad_param] = self.variables[param].grad

    def gradient(self, y: str = 'loss'):
        return list(self.grad_params.items())

    def add_output(self, output: str):
        self.outputs[output] = output

    def fetch_tensor(self, name):
        return self.fetch_tensors([name])[0]

    def fetch_tensors(self, names):
        tensors = []
        for each_name in names:
            var = self.variables[each_name]
            if var is None:
                print('Trying to fetch a None tensor', each_name)
                tensors.append(None)
                continue
            if self.cuda:
                var = var.cpu()
            if isinstance(var, autograd.Variable):
                var = var.data
            tensors.append(var.numpy())
        return tensors

    def fetch_internal_tensor(self, name):
        return self.fetch_internal_tensors([name])[0]

    def fetch_internal_tensors(self, names):
        return [self.variables.get(each_name) for each_name in names]

    def feed_tensor(self, name, new_value, device_option=None, is_param=False):
        if isinstance(new_value, np.ndarray):
            t = torch.from_numpy(new_value)
            requires_grad = is_param
            if self.cuda:
                new_value = autograd.Variable(t.cuda(), requires_grad=requires_grad)
            else:
                new_value = autograd.Variable(t, requires_grad=requires_grad)
        self.variables[name] = new_value

        if is_param:
            self.grad_params[name] = "grad_" + name


class PyTorchNativeNetwork(PyTorchNetwork):
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.outputs = set()

    @property
    def parameters(self):
        return self.module.parameters()

    def gradient(self, y: str = 'loss'):
        return [(p, p.grad) for p in self.module.parameters()]

    def fetch_tensors(self, tensors):
        result = []
        for tensor in tensors:
            var = tensor
            if var is None:
                print('Trying to fetch a None tensor')
                result.append(None)
                continue
            result.append(var.detach().cpu().numpy())
        return result

    def feed_tensor(self, param, new_value, device_option=None, is_param=False):
        with torch.no_grad():
            param[:] = torch.from_numpy(new_value)
