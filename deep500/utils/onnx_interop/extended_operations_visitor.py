import abc

from deep500.utils.onnx_interop.extended_operations import StopGradient, AllReduceGradientsOperation


class CustomOperationsVisitor(object, metaclass=abc.ABCMeta):
    def visit_stopgradient(self, stop_gradient: StopGradient, network):
        pass

        
class DistributedOperationsVisitor:
    def visit_allreduce_gradients(self, op: AllReduceGradientsOperation, network):
        pass

