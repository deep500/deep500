from typing import Dict, List

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core

import deep500 as d5

from .caffe2_network import Caffe2Network
from .caffe2_visitor_impl import Caffe2Visitor
from caffe2.proto.caffe2_pb2 import DeviceOption


class Caffe2GraphExecutor(d5.GraphExecutor):
    def __init__(self, model: d5.ops.OnnxModel, device_option: DeviceOption,
                 events: List[d5.ExecutorEvent] = []):
        super(Caffe2GraphExecutor, self).__init__(Caffe2Network(device_option), events)
        self.device_option = device_option
        self.model = model

        with core.DeviceScope(self.device_option):
            model.accept(Caffe2Visitor(device_option), self.network)

    def inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.network._setup()

        for event in self.events:
            event.before_executor(inputs)
        
        if self.network.device_option.device_type == caffe2_pb2.CUDA:
            self.network.test_model.net.RunAllOnGPU(use_cudnn=True)

        for node_name, value in inputs.items():
            self.network.workspace.FeedBlob(node_name, value, self.device_option)

        self.network.workspace.RunNetOnce(self.network.test_model.net)

        for key in self.network.output_dict.keys():
            self.network.output_dict[key] = self.network.workspace.FetchBlob(key)

        for event in self.events:
            event.after_inference(self.network.output_dict)
            
        return self.network.output_dict

    def inference_and_backprop(self, inputs: Dict[str, np.ndarray], y: str = 'loss') -> Dict[str, np.ndarray]:
        self.network._setup()
        self.network.add_loss(y)

        for event in self.events:
            event.before_executor(inputs)
        
        if self.device_option.device_type == caffe2_pb2.CUDA:
            self.network.train_model.param_init_net.RunAllOnGPU(use_cudnn=True)
            self.network.train_model.net.RunAllOnGPU(use_cudnn=True)

        for node_name, value in inputs.items():
            self.network.workspace.FeedBlob(node_name, value, self.device_option)

        self.network.workspace.RunNetOnce(self.network.train_model.net)

        for key in self.network.output_dict.keys():
            self.network.output_dict[key] = self.network.workspace.FetchBlob(key)

        for event in self.events:
            event.after_backprop(self.network.output_dict)
                        
        return self.network.output_dict
