"""
Abstract class for optimizers and stochastic optimizers. 
"""
import abc
import numpy as np
from typing import Any, Callable, Dict, List

from deep500.lv1.graph_executor import GraphExecutor
from deep500.lv2.event import OptimizerEvent, StopTraining
from deep500.lv2.sampler import Sampler


class Optimizer(metaclass=abc.ABCMeta):
    """ Abstraction for optimization (e.g., Stochastic Gradient Descent) on a 
        graph executor. """

    def __init__(self, executor: GraphExecutor, loss: str = 'loss',
                 gradient_modifier: Callable[[str, Any, Any], Any] = None):
        """ Initializes an abstract optimizer.
            @param executor Graph executor to use for training.
            @param loss Node name of parameter to optimize.
            @param gradient_modifier Modifies gradients after computations
                                     and before the optimizer step (e.g.,
                                     for regularization). Receives 
                                     (name, param, gradient) as input and
                                     returns the new gradient as output.
        """
        self.executor = executor
        self.network = executor.network
        self.loss = loss
        self.gradient_modifier = gradient_modifier

    def train(self, iterations: int, sampler: Sampler, 
              events: List[OptimizerEvent] = []) -> np.ndarray:
        """ An entire training procedure.
            @param iterations Maximum amount of iterations to train for.
            @param sampler The Sampler object to use for obtaining data.
            @param loss The name of the node to optimize.
            @param events List of TrainingEvent objects to use for training
            @return Final loss node value.
        """
        raise NotImplementedError

    def step(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ Perform one step in the training procedure (not always possible).
            @param inputs Input dictionary to feed in to the process.
            @param events List of TrainingEvent objects to use for training
            @return Outputs from executor's inference_and_backprop method.
        """
        raise ValueError('Single-stepping not supported')

    def set_parameter(self, name: str, value: Any,
                      predicate: Callable[[Any], bool] = None):
        """ Set an optimizer hyperparameter (e.g., learning rate)
            before or during optimization.
            @param name: Hyperparameter name.
            @param value: New value to set.
            @param predicate: An optional predicate that receives the
                              object to set, and returns True if it should be
                              changed or False otherwise. Can be used to set
                              different values for parameter groups.
        """
        raise NotImplementedError

    def as_operator(self):
        """ Returns a CustomOperator that runs an optimizer step.
        """
        raise NotImplementedError


class FirstOrderOptimizer(Optimizer, metaclass=abc.ABCMeta):
    """ An optimizer that loops in steps when training. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, iterations: int, sampler: Sampler,
              events: List[OptimizerEvent] = []) -> np.ndarray:
        """ An entire training procedure using the three-step optimizer.
            @param iterations Maximum amount of iterations to train for.
            @param sampler The Sampler object to use for obtaining data.
            @param events List of TrainingEvent objects to use for training.
            @return Final loss node value.
        """
        out = {}

        if iterations == 0:
            def _infloop(): # Loop indefinitely
                while True:
                    yield None
            rng = _infloop()
        else:
            rng = range(iterations)            

        try:
            for t in rng:
                sample = sampler() # Get sample

                for event in events:
                    event.before_optimizer_step(self.executor, self, sample)

                out = self.step(sample)

                for event in events:
                    event.after_optimizer_step(self.executor, self, out,
                                               out[self.loss])
        except StopIteration:
            # Dataset/sampler indicates that the set has ended
            pass
        except StopTraining:
            # Events requested training to be stopped
            raise

        return out[self.loss]


class ThreeStepOptimizer(FirstOrderOptimizer, metaclass=abc.ABCMeta):
    """ An optimizer that applies updates in three steps, enabling distributed
        optimization. """

    def __init__(self, executor: GraphExecutor, loss: str = 'loss'):
        super(ThreeStepOptimizer, self).__init__(executor, loss)

        
    def step(self, inputs):
        self.executor.network.setup()
        self.new_input()
        [self.prepare_param(param) for param in self.executor.network.get_params()]
        outputs = self.executor.inference_and_backprop(inputs, self.loss)
        gradients = self.executor.network.gradient(self.loss)
        for (param_name, grad_name) in gradients:
            param, grad = self.executor.network.fetch_tensors([param_name, grad_name])
            if self.gradient_modifier is not None:
                grad = self.gradient_modifier(grad_name, param, grad)
            param = self.update_rule(grad, param, param_name)
            self.executor.network.feed_tensor(param_name, param, is_param=True)

        return outputs

    def new_input(self):
        """
        This function is called before the next inference round so that any initialization / update can be done.
        Most optimizers don't need this, for usage example see the Adam optimizer.
        """
        pass

    def prepare_param(self, param_name):
        """
        Called before inference, in order to modify a parameter (e.g., in Nesterov's accelerated gradient descent).
        """
        pass        
        
    def update_rule(self, grad: np.ndarray, old_param: np.ndarray, param_name: str):
        """
        Given gradients and old parameters, computes a new parameter value.
        """
        pass


class UpdateRuleOptimizer(ThreeStepOptimizer, metaclass=abc.ABCMeta):
    """ An optimizer that only applies an update rule after computing gradients. """
    
    def step(self, inputs):
        self.executor.network.setup()
        outputs = self.executor.inference_and_backprop(inputs, self.loss)
        gradients = self.executor.network.gradient(self.loss)
        for (param_name, grad_name) in gradients:
            param, grad = self.executor.network.fetch_tensors([param_name, grad_name])
            if self.gradient_modifier is not None:
                grad = self.gradient_modifier(grad_name, param, grad)
            param = self.update_rule(grad, param, param_name)
            self.executor.network.feed_tensor(param_name, param, is_param=True)

        return outputs
