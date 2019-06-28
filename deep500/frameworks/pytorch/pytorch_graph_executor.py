from typing import Dict, List

import numpy as np
import torch.nn

import deep500 as d5

from .pytorch_network import PyTorchNetwork, PyTorchNativeNetwork
from .pytorch_visitor import PyTorchMetaVisitor, PyTorchVisitor


class PyTorchGraphExecutor(d5.GraphExecutor):

    def __init__(self, model: d5.ops.OnnxModel, device: d5.DeviceType,
                 events: List[d5.ExecutorEvent] = []):
        super().__init__(model, events)
        self.model = model

        self.setup_done = False
        self.network = PyTorchNetwork(device)

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

    def inference_and_backprop_internal(self, inputs: Dict[str, np.ndarray],
                                        loss: str):
        self.network._feed_input(inputs)
        self.model.accept(PyTorchVisitor(), self.network)
        loss = self.network.variables[loss]
        loss.backward()
        return loss


class PyTorchNativeGraphExecutor(PyTorchGraphExecutor):
    def __init__(self, module: torch.nn.Module, loss: torch.nn.Module,
                 input_node_name='0',
                 output_node_name='output', label_node_name='label',
                 loss_node_name='loss',
                 events: List[d5.ExecutorEvent] = [],
                 device: d5.DeviceType = None, with_outputs = False):
        """ Creates a graph executor of an existing TF graph.
            @param loss_node A Tensorflow Tensor or Operation object.
            @param session An existing Tensorflow session.
            @param events A list of events to invoke.
        """
        # Do not call super() here!
        self.network = PyTorchNativeNetwork(module)
        self.devname = 'cuda' if device is None or device.is_gpu() else 'cpu'
        self.events = events
        self.model = module.to(self.devname)
        self.is_training = True
        self.loss = loss.to(self.devname) if loss is not None else None
        self.innode = input_node_name
        self.outnode = output_node_name
        self.labelnode = label_node_name
        self.lossnode = loss_node_name
        self.with_outputs = with_outputs
        self.network.outputs = []

    def setup(self):
        pass

    def inference(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)

        if self.is_training is True:
            self.model.eval()
            self.is_training = False
        output = self.model(torch.from_numpy(input[self.innode]).to(self.devname))
        if self.loss is not None:
            y_tensor = torch.tensor(input[self.labelnode], dtype=torch.long,
                                    device=self.devname)
            loss = self.loss(output, y_tensor)
            outputs = {self.outnode: output.detach().cpu().numpy(),
                       self.lossnode: loss.detach().cpu().numpy()}
        else:
            outputs = {self.outnode: output.detach().cpu().numpy()}

        for event in self.events:
            event.after_inference(outputs)

        return outputs

    def inference_and_backprop(self, input: Dict[str, np.ndarray],
                               y: str = 'loss') -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)

        if self.is_training is False:
            self.model.train()
            self.is_training = True

        self.model.zero_grad()
        y_tensor = torch.tensor(input[self.labelnode], dtype=torch.long,
                                device=self.devname)
        output = self.model(torch.from_numpy(input[self.innode]).to(self.devname))
        loss = self.loss(output, y_tensor)
        loss.backward()

        outputs = {self.outnode: output.detach().cpu().numpy(),
                   y: loss.detach().cpu().numpy()}
        if self.with_outputs:
            outputs.update({('%s_grad' % p): p.grad.detach().numpy()
                            for p in self.model.parameters()})

        for event in self.events:
            event.after_backprop(outputs)

        return outputs

    def inference_and_backprop_internal(self, inputs: Dict[str, np.ndarray],
                                        loss: str):
        if self.is_training is False:
            self.model.train()
            self.is_training = True

        self.model.zero_grad()
        y_tensor = torch.tensor(inputs[self.labelnode], dtype=torch.long,
                                device=self.devname)
        output = self.model(torch.from_numpy(inputs[self.innode]).to(self.devname))
        loss = self.loss(output, y_tensor)
        loss.backward()
        return loss
