from typing import Dict, List

import numpy as np
import torch.nn

import deep500 as d5

from .pytorch_network import PyTorchNetwork, PyTorchNativeNetwork
from .pytorch_visitor import PyTorchVisitor


class PyTorchGraphExecutor(d5.GraphExecutor):

    def __init__(self, model: d5.ops.OnnxModel, device: d5.DeviceType,
                 events: List[d5.ExecutorEvent] = []):
        super().__init__(model, events)
        self.devname = 'cuda' if device is None or device.is_gpu() else 'cpu'
        self.network = PyTorchNetwork(device)
        self.visitor = PyTorchVisitor()
        self.is_training = False
        model.accept(self.visitor, self.network)
        self.model = self.visitor.model.to(self.devname)
        self.model.eval()
        new_network = PyTorchNativeNetwork(self.model)
        new_network.outputs = self.network.outputs
        self.network = new_network
        torch.cuda.empty_cache()

    def inference(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)

        if self.is_training is True:
            self.model.eval()
            self.is_training = False

        for name, val in input.items():
            self.model._params[name] = torch.tensor(val).to(self.devname)

        with torch.no_grad():
            self.model()

        output = {}
        for i, out in enumerate(list(self.network.outputs)):
            output[out] = self.model._params[out].detach().cpu().numpy()

        for event in self.events:
            event.after_inference(output)
            
        return output

    def inference_and_backprop(self, input: Dict[str, np.ndarray], y: str = 'loss') -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)

        self.model.zero_grad()
        self.inference_and_backprop_internal(input, y)

        output = {}
        for i, out in enumerate(list(self.network.outputs)):
            output[out] = self.model._params[out].detach().cpu().numpy()

        # Add gradients
        output.update({pname: p.grad for pname, p
                       in self.model.named_parameters()})

        for event in self.events:
            event.after_backprop(output)
            
        return output

    def inference_and_backprop_internal(self, inputs: Dict[str, np.ndarray],
                                        loss: str):
        if self.is_training is False:
            self.model.train()
            self.is_training = True

        for name, val in inputs.items():
            self.model._params[name] = torch.tensor(val).to(self.devname)

        loss_ = self.model()
        loss_.backward()

        return loss_


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
