import abc
import contextlib
from typing import List, Tuple

import onnx

from deep500.utils.device import DeviceType, GPUDevice
from deep500.utils.onnx_interop import parser
from deep500.utils.onnx_interop.onnx_objects import OnnxModel, Operation

import numpy as np
from typing import Dict

"""
This is an abstract representation of the different networks. 
"""


class Network(metaclass=abc.ABCMeta):
    """ Networks are defined as a dependency DAG of named nodes (strings).
        A network has a set of inputs and outputs (for inference and training),
        it can be set up (every time it becomes "dirty"), tensors (node 
        contents) can be fed and fetched, and gradient nodes can be requested. 

        The internal definition of a network is implementation-dependent,
        and likely uses an existing representation (e.g., a TensorFlow Graph).
    """
    # TODO(talbn): General graph manipulation API
    def __init__(self):
        pass

    def setup(self):
        """
        Called if network is dirty before inference.
        """
        pass

    def teardown(self):
        """
        Call to close any open files and sessions
        """
        pass

    def gradient(self) -> Dict[str, str]:
        """
        Calculates the gradient on this network
        """
        pass

    def fetch_tensors(self, names):
        """
        Fetches numpy tensors for given names
        """
        pass

    def feed_tensor(self, name, new_value, device_option=None, is_param=False):
        """
        Feed the given tensor to the workspace of the network
        @param name name of the tensor
        @param new_value new value of the tensor
        @param device_option specify device to put value on
        @param create_as_param save this as a trainable parameter
        """
        pass

    def add_output(self, name: str):
        """
        Define some additional network output you want to get back via the inference method
        @param name node name you want to get back the result of
        """
        pass

    def get_params(self):
        """
        @return all the names of the trainable variables
        """
        pass

    def get_input_nodes(self) -> List[str]:
        """
        @return all the names of the input (source) nodes
        """
        pass

    def get_output_nodes(self) -> List[str]:
        """
        @return all the names of the output (sink) nodes
        """
        pass
