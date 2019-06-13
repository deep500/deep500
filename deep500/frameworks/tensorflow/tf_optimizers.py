import tensorflow as tf
from typing import Union

from deep500.lv2.optimizer import FirstOrderOptimizer
from .tf_graph_executor import TensorflowGraphExecutor


class TFOptimizer(FirstOrderOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str):
        if not isinstance(executor, TensorflowGraphExecutor):
            raise TypeError('TensorFlow optimizer must use TensorFlow executor')
        super().__init__(executor, loss)
        self.minimize = None

    def as_operator(self):
        return self.op.minimize(self.network.fetch_internal_tensor(self.loss))

    def _fetch_or_constant(self, value: Union[str, float]):
        if isinstance(value, str):
            return self.network.fetch_internal_tensor(value)
        return tf.Variable(value, trainable=False)

    def set_parameter(self, name, value, predicate=None):
        if hasattr(self, name):
            var: tf.Variable = getattr(self, name)
            if predicate is None or predicate(var):
                var.assign(value)

    def step(self, inputs):
        # Lazy initialize operator
        if self.minimize is None:
            self.minimize = self.as_operator()
            # Force re-initialization of network to init optimizer
            self.network.vars_initialized = False

        return self.executor.fetch(inputs, self.minimize, self.loss)


class GradientDescent(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.1):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.op = tf.train.GradientDescentOptimizer(learning_rate=self.lr)


class AdamOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float] = 0.001,
                 beta1: Union[str, float] = 0.9,
                 beta2: Union[str, float] = 0.999,
                 epsilon: Union[str, float] = 1e-08):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.beta1 = self._fetch_or_constant(beta1)
        self.beta2 = self._fetch_or_constant(beta2)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                         beta1=self.beta1, beta2=self.beta2,
                                         epsilon=self.epsilon)


class RMSPropOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 decay: Union[str, float] = 0.9,
                 momentum: Union[str, float] = 0.0,
                 epsilon: Union[str, float] = 1e-10):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.decay = self._fetch_or_constant(decay)
        self.momentum = self._fetch_or_constant(momentum)
        self.epsilon = self._fetch_or_constant(epsilon)
        self.op = tf.train.RMSPropOptimizer(learning_rate=self.lr, 
                                            decay=self.decay, 
                                            momentum=self.momentum, 
                                            epsilon=self.epsilon)


class AdaGradOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str,
                 learning_rate: Union[str, float],
                 initial_accumulator_value: Union[str, float] = 0.1):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.iav = self._fetch_or_constant(initial_accumulator_value)
        self.op = tf.train.AdaGradOptimizer(learning_rate=self.lr, 
                                            initial_accumulator_value=self.iav)


class MomentumOptimizer(TFOptimizer):
    def __init__(self, executor: TensorflowGraphExecutor, loss: str, 
                 learning_rate: Union[str, float],
                 momentum: Union[str, float]):
        super().__init__(executor, loss)
        self.lr = self._fetch_or_constant(learning_rate)
        self.momentum = self._fetch_or_constant(momentum)
        self.op = tf.train.MomentumOptimizer(learning_rate=self.lr, 
                                             momentum=self.momentum)
