from typing import Dict, List

import numpy as np

import deep500 as d5

import tensorflow as tf

from .tf_network import TensorflowNetwork
from .tf_visitor_impl import TensorflowVisitor


class TensorflowGraphExecutor(d5.GraphExecutor):
    def __init__(self, model: d5.ops.OnnxModel, device: d5.DeviceType, events: List[d5.ExecutorEvent] = []):
        super(TensorflowGraphExecutor, self).__init__(TensorflowNetwork(device), events)
        self.setup_done = False
        self.model = model

        model.accept(TensorflowVisitor(), self.network)

    def inference(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)
            
        output_list = self.network.fetch_internal_tensors(self.network.output_names)

        # map string to tensor input
        input_dict = {self.network.fetch_internal_tensor(name): v for (name, v) in input.items()}

        if not self.network.vars_initialized:
            init = tf.global_variables_initializer()

            self.network.get_normal_session().run(init)
            self.network.vars_initialized = True

        output = self.network.get_normal_session().run(output_list, input_dict)

        out_dict = {}
        for i, out in enumerate(self.network.output_names):
            out_dict[out] = output[i]

        for event in self.events:
            event.after_inference(out_dict)
        
        return out_dict

    def inference_and_backprop(self, input: Dict[str, np.ndarray], y: str = 'loss') -> Dict[str, np.ndarray]:
        for event in self.events:
            event.before_executor(input)
        
        loss = self.network.fetch_internal_tensor(y)

        # map string to tensor input
        input_dict = {self.network.fetch_internal_tensor(name): v for (name, v) in input.items()}
        output_list = self.network.fetch_internal_tensors(self.network.output_names)

        # Here we provide gradients in numpy form so that we can work on them externally
        params = list(self.network.variables.values())
        if y in self.network._gradients:
            grad = self.network._gradients[y]
        else:
            grad = tf.gradients(loss, params)
            self.network._gradients[y] = grad

        (grad_names, grad_tensors, _vars) = self.network.get_gradient_names(self.network._gradients[y], y)

        if not self.network.vars_initialized:
            init = tf.global_variables_initializer()
            self.network.get_normal_session().run(init)
            self.network.vars_initialized = True

        result = self.network.get_normal_session().run(fetches=[[loss], grad_tensors, output_list], feed_dict=input_dict)
        loss_np, grad_numpy, outputs = result

        self.network.numpy_by_grad = {v: grad_numpy[i] for i, v in enumerate(grad_tensors)}

        for event in self.events:
            event.after_backprop(self.network.numpy_by_grad)
        
            
        result = {y: loss_np}
        result.update({k.name: v for k,v in self.network.numpy_by_grad.items()})
        result.update({k: v for k,v in zip(self.network.output_names, outputs)})            

        return result

    def fetch(self, input: Dict[str, np.ndarray], op, y: str = 'loss') -> Dict[str, np.ndarray]:
        loss = self.network.fetch_internal_tensor(y)

        # map string to tensor input
        input_dict = {self.network.fetch_internal_tensor(name): v for (name, v) in input.items()}
        output_list = self.network.fetch_internal_tensors(self.network.output_names)

        if not self.network.vars_initialized:
            init = tf.global_variables_initializer()
            self.network.get_normal_session().run(init)
            self.network.vars_initialized = True

        result = self.network.get_normal_session().run(fetches=[[op], [loss], output_list], feed_dict=input_dict)
        _, loss_np, outputs = result
            
        result = {y: loss_np}
        result.update({k: v for k,v in zip(self.network.output_names, outputs)})            

        return result
