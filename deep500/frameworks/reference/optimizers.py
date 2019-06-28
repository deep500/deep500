import abc
import numpy as np

import deep500 as d5


class ReferenceOptimizer(d5.Optimizer, abc.ABC):
    def set_parameter(self, name, value, predicate=None):
        if hasattr(self, name):
            if predicate is None or predicate(getattr(self, name)):
                setattr(self, name, value)

    def as_operator(self):
        # TODO: Implement
        return None


class GradientDescent(d5.UpdateRuleOptimizer, ReferenceOptimizer):

    def __init__(self, executor: d5.GraphExecutor, loss: str = 'loss', lr=0.1, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = lr

    def update_rule(self, grad, old_param, param_name):
        return old_param - self.lr * grad


class MomentumOptimizer(d5.UpdateRuleOptimizer, ReferenceOptimizer):

    def __init__(self, executor: d5.GraphExecutor, loss: str = 'loss', lr=0.01, momentum=0.1,
                 **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.accumulation = {}
        self.avg = 0
        self.n = 1

    def update_rule(self, grad, old_param, param_name):
        # accumulation = momentum * accumulation + gradient
        
        acc = self.momentum * self.get_accumulation(param_name) + grad
        self.accumulation[param_name] = acc

        # param = param - learning_rate * accumulation
        return old_param - self.lr * acc

    def get_accumulation(self, grad_name):
        if self.accumulation.get(grad_name) is None:
            self.accumulation[grad_name] = 0
        return self.accumulation[grad_name]


class AdamOptimizer(d5.ThreeStepOptimizer, ReferenceOptimizer):

    def __init__(self, executor: d5.GraphExecutor, loss: str = 'loss', lr=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-08, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_momentum = {}
        self.second_momentum = {}
        self.t = 0

    def new_input(self):
        self.t += 1

    def update_rule(self, grad, old_param, param_name):
        # update biased first moment estimate
        first_moment = self.beta1 * self.get_first_momentum(param_name) + (1 - self.beta1) * grad
        self.first_momentum[param_name] = first_moment
        # update biased second raw moment estimate
        second_moment = self.beta1 * self.get_second_momentum(param_name) \
                        + (1 - self.beta2) * np.power(grad, 2)
        self.second_momentum[param_name] = second_moment
        # compute bias - corrected  first moment estimate
        corrected_first_moment = first_moment / (1 - self.beta1 ** self.t)
        # compute bias corrected second raw moment estimate
        corrected_second_moment = second_moment / (1 - self.beta2 ** self.t)
        # update parameters
        new_param = old_param - self.lr * corrected_first_moment / (
            np.sqrt(corrected_second_moment + self.epsilon))
        return new_param

    def get_first_momentum(self, param_name):
        if self.first_momentum.get(param_name) is None:
            self.first_momentum[param_name] = 0
        return self.first_momentum[param_name]

    def get_second_momentum(self, param_name):
        if self.second_momentum.get(param_name) is None:
            self.second_momentum[param_name] = 0
        return self.second_momentum[param_name]


class AdaGradOptimizer(d5.UpdateRuleOptimizer, ReferenceOptimizer):
    def __init__(self, executor: d5.GraphExecutor, loss: str = 'loss', lr=1e-2,
                 eps=1e-6, **kwargs):
        super().__init__(executor, loss, **kwargs)
        self.lr = lr
        self.eps = eps
        self.squares = {}

    def _train(self, inputs):
        outputs = self.executor.network.inference_and_backprop(inputs)
        gradients = self.executor.network.gradient()
        for (param_name, grad_name) in gradients:
            param, grad = self.executor.network.fetch_tensors([param_name,
                                                               grad_name])
            param = self.new_param(grad, param, param_name)
            self.executor.network.feed_tensor(param_name, param)

        return outputs

    def new_param(self, grad, old_param, param_name):
        squared_grad = self.squares[param_name] if param_name in self.squares \
            else np.zeros(grad.shape)
        squared_grad += np.linalg.norm(grad) ** 2
        self.squares[param_name] = squared_grad
        adjusted_lr = self.lr / (self.eps + np.sqrt(squared_grad))
        return old_param - adjusted_lr * grad
