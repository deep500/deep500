from typing import Dict, List, Tuple

from .reference_network import ReferenceNetwork
from networkx.algorithms.dag import topological_sort

import numpy as np
import deep500 as d5
import networkx as nx

class ReferenceGraphExecutor(d5.GraphExecutor):
    def __init__(self, model: d5.ops.OnnxModel, device: d5.DeviceType, events: List[d5.ExecutorEvent] = [],
                 use_python_ops=False):
        """
        Creates a reference operator graph executor. Can use Python reference implementations (very slow)
        or compiled C++ reference implementations.
        @param model The model to build the network from
        @param device The device to use
        @param events Event objects to invoke
        @param use_python_ops If True, uses Python reference implementations, otherwise compiles and runs
                              C++ implementations
        """
        super(ReferenceGraphExecutor, self).__init__(ReferenceNetwork(device), events)
        if isinstance(device, d5.GPUDevice):
            print('Warning: GPU reference operators are currently not implemented. Falling back to CPU')
        self.model = model
        self.use_python_ops = use_python_ops

    def inference(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.network.train_mode = False
        
        for event in self.events:
            event.before_executor(input_dict)
        
        if not self.network.built_graph:
            # Choose visitor according to settings
            if self.use_python_ops:
                from .reference_build_graph_visitor_impl import ReferenceBuildGraphVisitor
            else:
                from .reference_build_graph_visitor_impl_cpp import ReferenceBuildGraphVisitor
                print('Compiling operators, this may take a while')
        
            # Build graph
            self.network.variables.update(input_dict)
            self.model.accept(ReferenceBuildGraphVisitor(), self.network)
            self.network.nodes_sorted_fwd = list(topological_sort(self.network.graph))
            self.network.nodes_sorted_bwd = list(reversed(list(topological_sort(self.network.graph))))
            self.network.built_graph = True

        out_dict = self.network.inference(input_dict)
        
        for event in self.events:
            event.after_inference(out_dict)
        
        self.network.output_dict.update(out_dict)

        return out_dict

    def inference_and_backprop(self, input_dict: Dict[str, np.ndarray], y: str= 'loss') -> Dict[str, np.ndarray]:
        self.network.train_mode = True
        
        for event in self.events:
            event.before_executor(input_dict)
        
        if not self.network.built_graph:
            # Choose visitor according to settings
            if self.use_python_ops:
                from .reference_build_graph_visitor_impl import ReferenceBuildGraphVisitor
            else:
                from .reference_build_graph_visitor_impl_cpp import ReferenceBuildGraphVisitor
                print('Compiling operators, this may take a while')
        
            # Build graph
            self.network.variables.update(input_dict)
            self.model.accept(ReferenceBuildGraphVisitor(), self.network)
            self.network.nodes_sorted_fwd = list(topological_sort(self.network.graph))
            self.network.nodes_sorted_bwd = list(reversed(list(topological_sort(self.network.graph))))
            self.network.built_graph = True

        out_dict = self.network.inference_and_backprop(input_dict, y)

        for event in self.events:
            event.after_backprop(out_dict)
        
        return out_dict