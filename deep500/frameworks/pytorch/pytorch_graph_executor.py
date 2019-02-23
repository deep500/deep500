from typing import Dict, List

import numpy as np
import torch

import deep500 as d5

from .pytorch_network import PyTorchNetwork
from .pytorch_visitor import PyTorchMetaVisitor, PyTorchVisitor

class PyTorchGraphExecutor(d5.GraphExecutor):

    def __init__(self, model: d5.ops.OnnxModel, events: List[d5.ExecutorEvent] = []):
        super().__init__(model, events)
        self.model = model

        # this stuff has to get resetted when teardown
        self.setup_done = False
        self.network = PyTorchNetwork()

    def setup(self):
        if not self.setup_done:
            self.model.accept(PyTorchMetaVisitor(), self.network)
            self.setup_done = True

    def inference(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.setup()

        for event in self.events:
            event.before_executor(input)

        
        self.network._feed_input(input)

        self.model.accept(PyTorchVisitor(), self.network)

        output = {}
        for i, out in enumerate(list(self.network.outputs.keys())):
            output[out] = self.network.fetch_tensor(out)

        for event in self.events:
            event.after_inference(output)
            
        return output

    def inference_and_backprop(self, input: Dict[str, np.ndarray], y: str = 'loss') -> Dict[str, np.ndarray]:
        self.setup()

        for event in self.events:
            event.before_executor(input)

        self.network._feed_input(input)
        self.model.accept(PyTorchVisitor(), self.network)

        # Zero grads
        for pname, p in self.network.variables.items():
            p.grad = None
        
        y_ = self.network.variables[y]
        y_.backward()
        self.network._save_gradients()
        
        output = {}
        for i, out in enumerate(list(self.network.outputs.keys())):
            output[out] = self.network.fetch_tensor(out)

        # Add gradients
        output.update({vname: self.network.fetch_tensor(vname) for vname in self.network.grad_params.values()})

        for event in self.events:
            event.after_backprop(output)
            
        return output
