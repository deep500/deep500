import torch
from typing import Callable, Union

from deep500.lv2.optimizer import FirstOrderOptimizer
from .pytorch_graph_executor import PyTorchGraphExecutor
from .pytorch_visitor import PyTorchVisitor


class PyTorchOptimizer(FirstOrderOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 with_outputs=True):
        if not isinstance(executor, PyTorchGraphExecutor):
            raise TypeError('PyTorch optimizer must use PyTorch executor')
        super().__init__(executor, loss)
        self.visitor = PyTorchVisitor()
        self.executor.setup()
        self.op = None
        self.with_outputs = with_outputs

    def _fetch_or_constant(self, value: Union[str, float]):
        if isinstance(value, str):
            return self.network.fetch_internal_tensor(value)
        return value

    def step(self, inputs):
        def closure():
            self.op.zero_grad()
            self.executor.network._feed_input(inputs)
            self.executor.model.accept(self.visitor, self.executor.network)
            loss = self.executor.network.variables[self.loss]
            loss.backward()
            return loss

        result = {self.loss: self.op.step(closure).item()}

        # Skip fetching all outputs if not necessary
        if self.with_outputs:
            result.update({o: tensor.cpu().detach().numpy() for o, tensor in
                           zip(self.network.outputs,
                               self.network.fetch_internal_tensors(
                                   self.network.outputs))})
        return result


class GradientDescent(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.1,
                 weight_decay: Union[str, float] = 0.0):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.SGD(self.executor.network.variables.values(),
                                  self.lr, weight_decay=self.decay)


class AdamOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.001,
                 beta1: Union[str, float] = 0.9,
                 beta2: Union[str, float] = 0.999,
                 epsilon: Union[str, float] = 1e-08,
                 weight_decay: Union[str, float] = 0.0):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.beta1 = self._fetch_or_constant(beta1)
        self.beta2 = self._fetch_or_constant(beta2)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.lr = self._fetch_or_constant(learning_rate)
        self.decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.Adam(self.executor.network.variables.values(), self.lr, (self.beta1, self.beta2),
                                   self.epsilon, self.decay)


class AdaGradOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 lr_decay: Union[str, float] = 0.0,
                 weight_decay: Union[str, float] = 0.0,
                 initial_accumulator_value: Union[str, float] = 0.1):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.iav = self._fetch_or_constant(initial_accumulator_value)
        self.lrdecay = self._fetch_or_constant(lr_decay)
        self.wdecay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.Adagrad(self.executor.network.variables.values(), self.lr,
                                      lr_decay=self.lrdecay, weight_decay=self.wdecay,
                                      initial_accumulator_value=self.iav)

class RMSPropOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 alpha: Union[str, float] = 0.99,
                 momentum: Union[str, float] = 0.0,
                 epsilon: Union[str, float] = 1e-8,
                 weight_decay: Union[str, float] = 0.0,
                 centered: bool = False):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.alpha = self._fetch_or_constant(alpha)
        self.decay = self._fetch_or_constant(weight_decay)
        self.momentum = self._fetch_or_constant(momentum)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = torch.optim.RMSprop(self.lr, self.alpha, self.epsilon,
                                      self.decay, self.momentum, centered)


class MomentumOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 momentum: Union[str, float],
                 dampening: Union[str, float] = 0.0,
                 weight_decay: Union[str, float] = 0.0,
                 nesterov: bool = False):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.momentum = self._fetch_or_constant(momentum)
        self.dampening = self._fetch_or_constant(dampening)
        self.decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.SGD(self.executor.network.variables.values(),
                                  self.lr, self.momentum, self.dampening,
                                  self.decay, nesterov=nesterov)

class LBFGSOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 1.0,
                 max_iter: int = 20,
                 max_eval: int = None,
                 tolerance_grad: Union[str, float] = 1e-5,
                 tolerance_change: Union[str, float] = 1e-9,
                 history_size: int = 100,
                 line_search_fn: Callable = None):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.tolerance_grad = self._fetch_or_constant(tolerance_grad)
        self.tolerance_change = self._fetch_or_constant(tolerance_change)
        self.op = torch.optim.LBFGS(self.executor.network.variables.values(),
                                    self.lr, max_iter, max_eval,
                                    self.tolerance_grad, self.tolerance_change,
                                    history_size, line_search_fn)