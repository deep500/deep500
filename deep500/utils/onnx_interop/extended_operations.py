""" Extends ONNX operators to support backpropagation and distributed training. """
from typing import Dict, List, Optional, Tuple
from deep500.utils.onnx_interop.onnx_objects import OnnxAttribute, Operation


class StopGradient(Operation):
    def __init__(
            self, input: List[str], output: List[str], name: Optional[str],
            op_type: Optional[str], domain: Optional[str],
            attributes: Dict[str, OnnxAttribute], doc_string: Optional[str]):
        super().__init__(input, output, name, op_type, domain, attributes,
                         doc_string)
        self.i_input = self.input[0]
        self.o_output = self.output[0]

    def accept(self, visitor, network):
        super(StopGradient, self).accept(visitor, network)
        visitor.visit_stopgradient(self, network)

    @classmethod
    def create_op(cls, i_input: str, o_output: str):
        attributes = {
        }
        return cls([i_input], [o_output], None, None, None, attributes, None)


class AllReduceGradientsOperation(Operation):
    def __init__(self, input: List[Tuple[str, str]],
                 comm_network):
        super().__init__(input, None, "AllReduceGradientOperation",
                         None, None, {}, None)
        self.gradients = self.input
        self.comm_network = comm_network

    def accept(self, visitor, network):
        visitor.visit_allreduce_gradients(self, network)

    @classmethod
    def create_op(cls, input: List[Tuple[str, str]],
                  comm_network):
        return cls(input, comm_network)


class ConsistentParameterServerGradientsOperation(Operation):
    def __init__(self, input: List[Tuple[str, str]],
                 comm_network):
        super().__init__(input, None,
                         "ConsistentParameterServerGradientsOperation",
                         None, None, {}, None)
        self.gradients = self.input
        self.comm_network = comm_network

    def accept(self, visitor, network):
        visitor.visit_consistent_parameter_server_gradients(self, network)

    @classmethod
    def create_op(cls, input: List[Tuple[str, str]],
                  comm_network):
        return cls(input, comm_network)


class InconsistentParameterServerGradientsOperation(Operation):
    def __init__(self, input: List[Tuple[str, str]],
                 comm_network):
        super().__init__(input, None,
                         "InconsistentParameterServerGradientsOperation",
                         None, None, {}, None)
        self.gradients = self.input
        self.comm_network = comm_network

    def accept(self, visitor, network):
        visitor.visit_inconsistent_parameter_server_gradients(self, network)

    @classmethod
    def create_op(cls, input: List[Tuple[str, str]],
                  comm_network):
        return cls(input, comm_network)


