import numpy as np
import torch
from torch import autograd, cuda, nn
from typing import List

import deep500 as d5


class PyTorchNetwork(d5.Network):
    def __init__(self):
        super(PyTorchNetwork, self).__init__()
        self.variables = {}
        self.inputs = {}
        self.outputs = {}
        self.optimizer_created = False
        self.grad_params = {}
        self.cuda = cuda.is_available()

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
        return [(param, param_grad) for (param, param_grad) in self.grad_params.items()]

    def add_output(self, output: str):
        self.outputs[output] = output

    def fetch_tensor(self, name):
        return self.fetch_tensors([name])[0]

    def fetch_tensors(self, names):
        tensors = []
        for each_name in names:
            var = self.variables.get(each_name)
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

    def fetch_tensor_internal(self, name):
        return self.fetch_tensors_internal([name])[0]

    def fetch_tensors_internal(self, names):
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
