import abc
from typing import Dict, List
import numpy as np

from deep500.lv1.network import Network
from deep500.lv1.event import ExecutorEvent

class GraphExecutor(metaclass=abc.ABCMeta):

    def __init__(self, network: Network, events: List[ExecutorEvent] = None):
        self.network = network
        self.events = events or []

    @abc.abstractmethod
    def inference(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Runs network inference for a given dictionary of node names to inputs.
        @param input A mapping from input node names to their values as numpy arrays.
        @return A mapping from output node names to their values as numpy arrays.
        """
        pass

    @abc.abstractmethod
    def inference_and_backprop(self, input: Dict[str, np.ndarray], y: str = 'loss') -> Dict[str, np.ndarray]:
        """
        Runs network inference and backpropagation for a given dictionary of inputs and the 
        node to compute gradients from.
        @param input A mapping from input node names to their values as numpy arrays.
        @param y The output node name for which to compute the gradient.
        @return A mapping from output node names to their values as numpy arrays.
        """
        pass
