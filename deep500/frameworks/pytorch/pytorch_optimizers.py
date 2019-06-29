import torch
from typing import Callable, Union

from deep500.lv2.optimizer import FirstOrderOptimizer
from .pytorch_graph_executor import PyTorchGraphExecutor


class PyTorchOptimizer(FirstOrderOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 with_outputs=True, **kwargs):
        if not isinstance(executor, PyTorchGraphExecutor):
            raise TypeError('PyTorch optimizer must use PyTorch executor')
        super().__init__(executor, loss, **kwargs)
        self.op = None
        self.with_outputs = with_outputs

    def _fetch_or_constant(self, value: Union[str, float]):
        if isinstance(value, str):
            return self.network.fetch_internal_tensor(value)
        return value

    def set_parameter(self, name, value, predicate=None):
        for param_group in self.op.param_groups:
            if name in param_group:
                if predicate is None or predicate(param_group):
                    param_group[name] = value

    def step(self, inputs):
        def closure():
            self.op.zero_grad()
            loss = self.executor.inference_and_backprop_internal(inputs,
                                                                 self.loss)

            # Modify gradients
            if self.gradient_modifier is not None:
                with torch.no_grad():
                    for pname, p in self.executor.model.named_parameters():
                        p.grad = self.gradient_modifier(pname, p, p.grad)

            return loss

        result = {self.loss: self.op.step(closure).item()}

        # Skip fetching all outputs if not necessary
        if self.with_outputs:
            for out in self.network.outputs:
                result[out] = self.executor.model._params[out].detach().cpu().numpy()

        return result


class GradientDescent(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.1,
                 weight_decay: Union[str, float] = 0.0, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.weight_decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.SGD(self.executor.network.parameters,
                                  self.lr, weight_decay=self.weight_decay)


class AdamOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.001,
                 beta1: Union[str, float] = 0.9,
                 beta2: Union[str, float] = 0.999,
                 epsilon: Union[str, float] = 1e-08,
                 weight_decay: Union[str, float] = 0.0, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.beta1 = self._fetch_or_constant(beta1)
        self.beta2 = self._fetch_or_constant(beta2)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.lr = self._fetch_or_constant(learning_rate)
        self.weight_decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.Adam(self.executor.network.parameters, self.lr,
                                   (self.beta1, self.beta2),
                                   self.epsilon, self.weight_decay)


class AdaGradOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 lr_decay: Union[str, float] = 0.0,
                 weight_decay: Union[str, float] = 0.0,
                 initial_accumulator_value: Union[str, float] = 0.1, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.initial_accumulator_value = \
            self._fetch_or_constant(initial_accumulator_value)
        self.lr_decay = self._fetch_or_constant(lr_decay)
        self.weight_decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.Adagrad(
            self.executor.network.parameters,
            self.lr,
            lr_decay=self.lr_decay,
            weight_decay=self.weight_decay,
            initial_accumulator_value=self.initial_accumulator_value)

class RMSPropOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 alpha: Union[str, float] = 0.99,
                 momentum: Union[str, float] = 0.0,
                 epsilon: Union[str, float] = 1e-8,
                 weight_decay: Union[str, float] = 0.0,
                 centered: bool = False, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.alpha = self._fetch_or_constant(alpha)
        self.weight_decay = self._fetch_or_constant(weight_decay)
        self.momentum = self._fetch_or_constant(momentum)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = torch.optim.RMSprop(self.lr, self.alpha, self.epsilon,
                                      self.weight_decay, self.momentum,
                                      centered)


class MomentumOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 momentum: Union[str, float],
                 dampening: Union[str, float] = 0.0,
                 weight_decay: Union[str, float] = 0.0,
                 nesterov: bool = False, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.momentum = self._fetch_or_constant(momentum)
        self.dampening = self._fetch_or_constant(dampening)
        self.weight_decay = self._fetch_or_constant(weight_decay)
        self.op = torch.optim.SGD(self.executor.network.parameters,
                                  self.lr, self.momentum, self.dampening,
                                  self.weight_decay, nesterov=nesterov)

class LBFGSOptimizer(PyTorchOptimizer):
    def __init__(self, executor: PyTorchGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 1.0,
                 max_iter: int = 20,
                 max_eval: int = None,
                 tolerance_grad: Union[str, float] = 1e-5,
                 tolerance_change: Union[str, float] = 1e-9,
                 history_size: int = 100,
                 line_search_fn: Callable = None, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = self._fetch_or_constant(learning_rate)
        self.tolerance_grad = self._fetch_or_constant(tolerance_grad)
        self.tolerance_change = self._fetch_or_constant(tolerance_change)
        self.op = torch.optim.LBFGS(self.executor.network.parameters,
                                    self.lr, max_iter, max_eval,
                                    self.tolerance_grad, self.tolerance_change,
                                    history_size, line_search_fn)
