import abc

from deep500.lv1.network import Network
from deep500.utils.onnx_interop.extended_operations_visitor import CustomOperationsVisitor, DistributedOperationsVisitor
from deep500.utils.onnx_interop.emptyoperations_visitor import EmptyOperationsVisitor
from deep500.utils.onnx_interop.loss_operations_visitor import LossOperationsVisitor
from deep500.utils.onnx_interop.onnx_objects import OnnxGraph, OnnxNode, OnnxModel, OnnxTensor, OnnxValueInfo
from deep500.utils.onnx_interop.operations_visitor import OperationsVisitor

"""
This extends the OperationsVisitor with a few more visit methods like graph and model. 
The visit_node gets called before each operation visit so if you have common functionality to every operation
put it there. 
"""


class OnnxModelVisitor:
    def visit_model(self, model: OnnxModel, network: Network):
        pass

    def visit_graph(self, graph: OnnxGraph, network: Network):
        pass

    def visit_node(self, node: OnnxNode, network: Network):
        """
        This method is called in the operation subclass so that common functionality
        to every node can be encapsulated here
        """
        pass

    def visit_net_input(self, input: OnnxValueInfo, network: Network):
        pass

    def visit_net_output(self, output: OnnxValueInfo, network: Network):
        pass

    def visit_initializer(self, each_initializer: OnnxTensor, network: Network):
        pass

    def visit_initializer_end(self, network: Network):
        """
        This method is called after all initializers have been called
        """
        pass

    def visit_graph_end(self, network: Network):
        """This method gets called after all nodes from a graph have been called"""
        pass


class OnnxBaseVisitor(OperationsVisitor,
                      LossOperationsVisitor,
                      CustomOperationsVisitor,
                      DistributedOperationsVisitor,
                      OnnxModelVisitor, metaclass=abc.ABCMeta):
    pass


class EmptyOnnxBaseVisitor(EmptyOperationsVisitor,
                           LossOperationsVisitor,
                           CustomOperationsVisitor,
                           DistributedOperationsVisitor,
                           OnnxModelVisitor, metaclass=abc.ABCMeta):
    pass
