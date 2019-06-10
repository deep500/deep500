import abc
from typing import List


class Loss(abc.ABC):
    def __init__(self, input: List[str], output: str):
        self.input = input
        self.output = output

    @abc.abstractmethod
    def accept(self, visitor, network):
        pass


class CrossEntropy(Loss):
    def __init__(self, input: List[str], output: str):
        super().__init__(input, output)
        self.i_X = input[0]
        self.i_target = input[1]
        self.o_output = output

    def accept(self, visitor, network):
        visitor.visit_cross_entropy(self, network)


class SoftmaxCrossEntropy(Loss):
    def __init__(self, input: List[str], output: str):
        super().__init__(input, output)
        self.i_X = input[0]
        self.i_target = input[1]
        self.o_output = output

    def accept(self, visitor, network):
        visitor.visit_softmax_cross_entropy(self, network)


class MeanSquaredError(Loss):
    def __init__(self, input: List[str], output: str):
        super().__init__(input, output)
        self.i_X = input[0]
        self.i_target = input[1]
        self.o_output = output

    def accept(self, visitor, network):
        visitor.visit_mean_squared_error(self, network)
