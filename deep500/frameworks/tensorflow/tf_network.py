from typing import List

import deep500 as d5


import tensorflow as tf


class TensorflowNetwork(d5.Network):
    def __init__(self, device_option: d5.DeviceType, verbose=False):
        super(TensorflowNetwork, self).__init__()
        self.device_option = device_option
        self.variables = {}
        self.tensors = {}
        self.output_names = []  # type: List[str]
        # holds outputs of last inference
        self.output_dict = {}
        # map (param_name, gradient_name)
        self.grad_names = {}
        # session
        self.session = None
        # variables initialized
        self.vars_initialized = False
        # possibility to add log writer to monitor session
        self.train_writer = None

        self.partial_handle = None
        self._gradients = {}
        self.hooks = []

        # saved gradient by variable
        self.numpy_by_grad = {}
        # gradient_names by loss
        self.gradient_names_by_y = {}

        tf_args = {}
        if verbose:
            tf_args['log_device_placement'] = True
        self.session_config = tf.ConfigProto(**tf_args) if self.device_option.is_gpu() \
            else tf.ConfigProto(device_count={'GPU': 0}, **tf_args)
        if self.device_option.is_gpu():
            self.session_config.gpu_options.visible_device_list = str(self.device_option.num)

        self.initializers = {}

    def _teardown(self):
        self.session.close()

    def get_monitored_session(self):
        if self.session is None:
            self.session = tf.train.MonitoredSession(
                session_creator=tf.train.ChiefSessionCreator(config=self.session_config), hooks=self.hooks)
        return self.session

    def get_normal_session(self):
        if self.session is None:
            self.session = tf.Session(config=self.session_config)
        return self.session

    def add_hooks(self, hooks):
        self.hooks.extend(hooks)

    def get_params(self):
        return list(self.variables.keys())

    def export_native(self, folder: str, inputs: List[str]):
        inputs = {k: self.fetch_internal_tensor(k) for k in inputs}
        outputs = {k: self.fetch_internal_tensor(k) for k in self.output_names}
        tf.saved_model.simple_save(self.session, folder, inputs, outputs)

    def _custom_parsing_context(self):
        dev_spec = tf.DeviceSpec(device_type=("GPU" if self.device_option.is_gpu() else "CPU"),
                                 device_index=self.device_option.num)
        return tf.device(dev_spec)

    def gradient(self, y: str = 'loss'):
        res = self.grad_names.get(y)
        if res is None:
            raise Exception('You have to call inference_and_backprop first with the correct y. Current y: {}'.format(y))
        return res
        
    def add_output(self, output: str):
        self.output_names.append(output)

    def get_gradient_names(self, gradients, y):
        names = []
        _vars = []
        for i, (name, var) in enumerate(list(self.variables.items())):
            if gradients[i] is None:  # TODO HBS: Check if we really all none is none
                continue
            names.append((name, gradients[i]))
            _vars.append(var)
        self.grad_names[y] = names
        return names, [g for g in gradients if g is not None], _vars

    def fetch_tensors(self, names):
        _names = [self.variables[name] if type(name) is str else name for name in names]
        result = []
        indices = []
        for i, name in enumerate(_names):
            if name in self.numpy_by_grad:
                result.append(self.numpy_by_grad[name])
            else:
                result.append(-1)
                indices.append(i)

        if not indices:
            return result
        fetch_vars = [_names[i] for i in indices if i != -1]
        output = self.get_normal_session().run(fetch_vars)

        j = 0
        out = []
        for i, res in enumerate(result):
            if type(res) == int:
                out.append(output[j])
                j += 1
            else:
                out.append(res)

        return out

    def fetch_tensor(self, name):
        return self.fetch_tensors([name])[0]

    def feed_tensor(self, name, new_value, device_option=None, is_param=False):
        if is_param and name not in self.variables:
            if name in self.tensors:
                var = self.tensors[name]
            else:
                var = tf.get_variable(name, initializer=new_value, trainable=True)
            self.variables[name] = var
        else:
            var = self.variables[name]
            var.load(new_value, self.session)

    def add_param(self, name: str, var: tf.Variable):
        self.variables[name] = var

    def fetch_variables(self, names):
        return [self.fetch_variable(name) for name in names]

    def fetch_variable(self, name):
        return self.variables.get(name)

    def feed_internal_tensor(self, name, tensor):
        self.tensors[name] = tensor

    def fetch_internal_tensor(self, name):
        if isinstance(name, (tf.Tensor, tf.Operation)):
            return name
        return self.tensors.get(name) if name in self.tensors else self.variables[name]

    def fetch_internal_tensors(self, names):
        return [self.fetch_internal_tensor(name) for name in names]
        
    def get_input_nodes(self) -> List[str]:
        graph = tf.get_default_graph()
        return [n.name + ':0' for n in graph.as_graph_def().node if len(n.input) == 0 and n.op == 'Placeholder']

    def get_output_nodes(self) -> List[str]:
        return self.output_names
