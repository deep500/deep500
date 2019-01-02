import os
from typing import List

from caffe2.proto import caffe2_pb2
from caffe2.proto.caffe2_pb2 import DeviceOption
from caffe2.python.modeling import initializers
from caffe2.python.predictor import mobile_exporter

from caffe2.python import core, workspace, model_helper, net_drawer

from deep500.lv1.network import Network


class Caffe2Network(Network):
    def __init__(self, device_option: DeviceOption):
        super(Caffe2Network, self).__init__()
        self.device_option = device_option

        self.train_model = model_helper.ModelHelper(name="train_default_net")
        self.test_model = model_helper.ModelHelper(name="test_default_net", init_params=False)
        self.train_net = self.train_model.net
        self.test_net = self.test_model.net
        self.train_init_net = self.train_model.param_init_net
        self.test_init_net = self.test_model.param_init_net
        self.workspace = workspace
        self.output_dict = {}
        self.param_names = None
        # dict that helps us remember that we already added the gradients to the graph for a given loss
        self.gradients_by_loss = {}
        self.is_cuda = (device_option.device_type == caffe2_pb2.CUDA)

    def _setup(self):
        self.workspace.RunNetOnce(self.train_init_net)
        self.workspace.RunNetOnce(self.test_init_net)

    def get_params(self):
        if self.param_names is None:
            self.param_names = [p._name for p in self.train_model.params]
        return self.param_names

    def _teardown(self):
        workspace.ResetWorkspace()

    def add_loss(self, y: str):
        if y != '' and y not in self.gradients_by_loss:
            self.gradients_by_loss[y] = y
            self.train_model.AddGradientOperators([y])

    def get_input_nodes(self):
        return list(self.input_dict.keys())
            
    def get_output_nodes(self):
        return list(self.output_dict.keys())

    def add_optimizer(self, optimizer):
        raise Exception('we do not use this')

    def gradient(self, y: str = 'loss'):
        gradients = self.train_model.param_to_grad.items()
        return gradients

    def _custom_parsing_context(self):
        return core.DeviceScope(self.device_option)

    def export_native(self, folder: str, inputs: List[str]):
        print("Save the model to init_net.pb and predict_net.pb")

        _, init_net2 = mobile_exporter.Export(workspace, self.train_model.param_init_net, [])
        init_net, predict_net = mobile_exporter.Export(workspace, self.train_model.net, self.train_model.params)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open("{}/train_init_net2.pb".format(folder), 'wb') as f:
            f.write(init_net2.SerializeToString())

        with open("{}/train_init_net.pb".format(folder), 'wb') as f:
            f.write(init_net.SerializeToString())

        with open("{}/train_predict_net.pb".format(folder), 'wb') as f:
            f.write(predict_net.SerializeToString())
        print("Saved the model to folder: {} and names train_predict_net.pb and train_init_net.pb".format(folder))

        init_net, predict_net = mobile_exporter.Export(workspace, self.test_model.net, self.test_model.params)
        with open("{}/test_init_net.pb".format(folder), 'wb') as f:
            f.write(init_net.SerializeToString())

        with open("{}/test_predict_net.pb".format(folder), 'wb') as f:
            f.write(predict_net.SerializeToString())
        print("Saved the model to folder: {} and names test_predict_net.pb and test_init_net.pb".format(folder))

    def teardown(self):
        pass

    def add_output(self, output: str):
        self.output_dict[output] = None

    def fetch_tensors(self, names):
        return [workspace.FetchBlob(name) for name in names]

    def feed_tensor(self, name, new_value, device_option=None, is_param=False):
        if is_param:
            self.train_model.create_param(name, new_value.shape,
                                          initializer=initializers.ExternalInitializer())
        device_option = device_option if device_option else self.device_option
        return workspace.FeedBlob(name, new_value, device_option)
