from typing import Dict, List, Tuple

from deep500.lv0.operators.operator_interface import CustomPythonOp
from deep500.lv0.operators.operator_interface import CustomCPPOp
from networkx.algorithms.dag import topological_sort

import numpy as np
import deep500 as d5
import networkx as nx

class ReferenceNetwork(d5.Network):
    def __init__(self, device_option: d5.DeviceType, verbose=False):
        super(ReferenceNetwork, self).__init__()
        self.device_option = device_option
        self.variables: Dict[str, np.ndarray] = {}
        self.output_names = []
        # holds outputs of last inference
        self.output_dict = {}
        # holds the names and values from the weights (learnable variables)
        self.gradients = {}

        self.tensors_op_inputs = {}
        self.tensors_op_outputs = {}
        self.graph = nx.DiGraph()
        self.built_graph = False
        self.nodes_sorted_fwd = []
        self.nodes_sorted_bwd = []

        self.gradients = {}
        self.grad_names = {}

        self.params = {}
        self.losses = {}


    def get_input_nodes(self) -> List[str]:
        return list(n for n in self.graph.nodes() if self.graph.in_degree(n) == 0)

    def get_output_nodes(self) -> List[str]:
        return self.output_names

    def get_params(self):
        return list(self.variables.keys())
        
    def gradient(self, y: str = 'loss'):
        res = self.grad_names.get(y)
        if res is None:
            raise Exception('You have to call inference_and_backprop first the correct y this y: {}'.format(y))
        return res
        
    def add_output(self, output: str):
        self.output_names.append(output)

    def inference(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.setup()

        self.variables.update(input_dict)

        nodes_custom_op = nx.get_node_attributes(self.graph, 'custom_op')
        nodes_operator = nx.get_node_attributes(self.graph, 'operator')
        
        for node_name in self.nodes_sorted_fwd:
            op = nodes_operator[node_name]
            custom_operator = nodes_custom_op[node_name]

            inputs = [self.variables[input_name] for input_name in op.input]

            print('start: forward computation of ' + str(node_name))
            if isinstance(custom_operator, CustomPythonOp):
                self.variables[op.output[0]] = custom_operator.forward(*inputs)
            elif isinstance(custom_operator, CustomCPPOp):
                custom_operator.forward(*inputs, self.variables[op.output[0]])
            else:
                raise ValueError('type of custom_operator is not supported')
            print('end: forward computation of ' + str(node_name))

        out_dict = {}
        for name in self.output_names:
            out_dict[name] = self.variables[name]
        return out_dict

    def inference_and_backprop(self, input_dict: Dict[str, np.ndarray], y: str= 'loss') -> Dict[str, np.ndarray]:
        self.setup()
        
        self.variables.update(input_dict)

        nodes_custom_op = nx.get_node_attributes(self.graph, 'custom_op')
        nodes_operator = nx.get_node_attributes(self.graph, 'operator')
        
        #inference
        for node_name in self.nodes_sorted_fwd:
            op = nodes_operator[node_name]
            custom_operator = nodes_custom_op[node_name]

            inputs = [self.variables[input_name] for input_name in op.input]

            print('start: forward computation of ' + str(node_name))
            if isinstance(custom_operator, CustomPythonOp):
                self.variables[op.output[0]] = custom_operator.forward(*inputs)
            elif isinstance(custom_operator, CustomCPPOp):
                custom_operator.forward(*inputs, self.variables[op.output[0]])
            else:
                raise ValueError('type of custom_operator is not supported')
            print('end: forward computation of ' + str(node_name))

        (custom_operator, op, node_name) = self.losses[y]
        print('start: forward computation of ' + str(node_name))
        if isinstance(custom_operator, CustomPythonOp):
            self.variables[op.output] = custom_operator.forward(*inputs)
        elif isinstance(custom_operator, CustomCPPOp):
            custom_operator.forward(*inputs, self.variables[op.output])
        else:
            raise ValueError('type of custom_operator is not supported')
        print('end: forward computation of ' + str(node_name))

        out_dict = {y: self.variables[y]}

        #backprop
        inputs = [self.variables[input_name] for input_name in op.input]
        outputs = [self.variables[y]]
        grad_in = np.ones_like(self.variables[y])

        (custom_operator, op, node_name) = self.losses[y]
        print('start: backward computation of ' + str(node_name))
        if isinstance(custom_operator, CustomPythonOp):
            grad_out = custom_operator.backward(grad_in, inputs, outputs)
        elif isinstance(custom_operator, CustomCPPOp):
            custom_operator.backward(grad_in, inputs, outputs, grad_out)
        else:
            raise ValueError('type of custom_operator is not supported')
        self.gradients[op.input[0]] = grad_out[0]
        print('end: backward computation of ' + str(node_name))

        for node_name in self.nodes_sorted_bwd:
            op = nodes_operator[node_name]
            custom_operator = nodes_custom_op[node_name]

            inputs = [self.variables[input_name] for input_name in op.input]
            outputs = [self.variables[output_name] for output_name in op.output]
            grad_in = [self.gradients[output_name] for output_name in op.output]
            grad_out = [self.gradients[input_name] for input_name in op.input]

            print('start: backward computation of ' + str(node_name))
            if isinstance(custom_operator, CustomPythonOp):
                grad_out = custom_operator.backward(grad_in, inputs, outputs)
                for i in range(len(op.input)):
                    self.gradients[op.input[i]] = grad_out[i]
            elif isinstance(custom_operator, CustomCPPOp):
                #for i in range(len(op.input)):
                    #grad_out = [self.gradients[j] for j in op.input]
                custom_operator.backward(*grad_in, *inputs, *outputs, *grad_out)
                for i in range(len(op.input)):
                    self.gradients[op.input[i]] = grad_out[i]
            else:
                raise ValueError('type of custom_operator is not supported')
            print('end: backward computation of ' + str(node_name))

        # having calculated the gradients we fill them in the variables dict
        # so that outsiders can load and check it
        for name in self.gradients.keys():
            self.variables[name + "_grad"] = self.gradients[name]

        return out_dict

    def get_gradient_names(self, gradients, y):
        names = []
        _vars = []
        for i, (name, var) in enumerate(list(self.variables.items())):
            if gradients[i] is None:  # TODO HBS: Check if we really all none is none
                continue
            names.append((name, gradients[i]))
            _vars.append(var)
        self.grad_names[y] = names
        return names, [g for g in gradients if g is not None], _vars

    def fetch_tensors(self, names):
        res = []
        for name in names:
            if name in self.params:
                res.append(self.params[name])
            else:
                res.append(self.variables[name])
        return res

    def fetch_tensor(self, name):
        return self.fetch_tensors([name])[0]

    def feed_tensor(self, name, value, is_param=False):
        if name in self.params or is_param:
            self.params[name] = value
            self.variables[name] = value
        else:
            self.variables[name] = value

    def _teardown(self):
        print('teardown')

    def fetch_variables(self, names):
        return [self.fetch_variable(name) for name in names]

    def fetch_variable(self, name):
        return self.variables.get(name)

    def feed_internal_tensor(self, name, tensor):
        self.variables[name] = tensor

    def fetch_internal_tensor(self, name):
        return self.variables[name]

    def fetch_internal_tensors(self, names):
        return [self.fetch_internal_tensor(name) for name in names]
