import abc
from deep500.lv1.network import Network
from deep500.utils.onnx_interop.losses import CrossEntropy, LabelCrossEntropy, MeanSquaredError


class LossOperationsVisitor(metaclass=abc.ABCMeta):

    def visit_cross_entropy(self, cross_entroy: CrossEntropy, network: Network):
        pass

    def visit_label_cross_entropy(self, label_cross_entropy: LabelCrossEntropy, network: Network):
        pass

    def visit_mean_squared_error(self, op: MeanSquaredError, network: Network):
        pass
