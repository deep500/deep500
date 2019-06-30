import tensorflow as tf
from typing import Union

from deep500.lv2.optimizer import FirstOrderOptimizer
from .tf_graph_executor import TensorflowGraphExecutor


class TFOptimizer(FirstOrderOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str, **kwargs):
        if not isinstance(executor, TensorflowGraphExecutor):
            raise TypeError('TensorFlow optimizer must use TensorFlow executor')
        super().__init__(executor, loss, **kwargs)
        self.minimize = self.as_operator()

    def as_operator(self, global_step=None):
        grads_and_vars = self.op.compute_gradients(
            self.network.fetch_internal_tensor(self.loss))

        # Modify gradients if necessary
        if self.gradient_modifier is not None:
            for i, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[i] = (self.gradient_modifier(grad.name,
                                                                var, grad), var)

        return self.op.apply_gradients(grads_and_vars, global_step)

    def _fetch_or_constant(self, value: Union[str, float]):
        if isinstance(value, str):
            return self.network.fetch_internal_tensor(value)
        return tf.Variable(value, trainable=False)

    def set_parameter(self, name, value, predicate=None):
        if hasattr(self, name):
            var: tf.Variable = getattr(self, name)
            if predicate is None or predicate(var):
                self.executor.session.run(var.assign(value))

    def step(self, inputs):
        result = self.executor.fetch(inputs, self.minimize, self.loss,
                                     is_training=True)
        return result


class GradientDescent(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.1, **kwargs):
        self.lr = self._fetch_or_constant(learning_rate)
        self.op = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        super().__init__(executor, loss, **kwargs)


class AdamOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.001,
                 beta1: Union[str, float] = 0.9,
                 beta2: Union[str, float] = 0.999,
                 epsilon: Union[str, float] = 1e-08, **kwargs):
        self.lr = self._fetch_or_constant(learning_rate)
        self.beta1 = self._fetch_or_constant(beta1)
        self.beta2 = self._fetch_or_constant(beta2)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                         beta1=self.beta1, beta2=self.beta2,
                                         epsilon=self.epsilon)
        super().__init__(executor, loss, **kwargs)


class RMSPropOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 decay: Union[str, float] = 0.9,
                 momentum: Union[str, float] = 0.0,
                 epsilon: Union[str, float] = 1e-10, **kwargs):
        self.lr = self._fetch_or_constant(learning_rate)
        self.decay = self._fetch_or_constant(decay)
        self.momentum = self._fetch_or_constant(momentum)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = tf.train.RMSPropOptimizer(learning_rate=self.lr, 
                                            decay=self.decay, 
                                            momentum=self.momentum, 
                                            epsilon=self.epsilon)
        super().__init__(executor, loss, **kwargs)


class AdaGradOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 initial_accumulator_value: Union[str, float] = 0.1, **kwargs):
        self.lr = self._fetch_or_constant(learning_rate)
        self.iav = self._fetch_or_constant(initial_accumulator_value)
        self.op = tf.train.AdaGradOptimizer(learning_rate=self.lr, 
                                            initial_accumulator_value=self.iav)
        super().__init__(executor, loss, **kwargs)


class MomentumOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str, 
                 learning_rate: Union[str, float],
                 momentum: Union[str, float], **kwargs):
        self.lr = self._fetch_or_constant(learning_rate)
        self.momentum = self._fetch_or_constant(momentum)
        self.op = tf.train.MomentumOptimizer(learning_rate=self.lr, 
                                             momentum=self.momentum)
        super().__init__(executor, loss, **kwargs)
