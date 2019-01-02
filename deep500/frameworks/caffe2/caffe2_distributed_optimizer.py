from caffe2.proto import caffe2_pb2
from caffe2.python import core, dyndep

from deep500.lv2.optimizer import Optimizer
from deep500.frameworks.caffe2.caffe2_network import Caffe2Network
from deep500.lv3.communication import CommunicationNetwork

custom_ops_loaded = False


def _load_custom_dll():
    global custom_ops_loaded
    if not custom_ops_loaded:
        import os
        lib_name = 'libcaffe2_mpi_operations.so'
        CUSTOM_OPERATIONS = 'CUSTOM_OPERATIONS'
        if CUSTOM_OPERATIONS not in os.environ:
            raise Exception('Set the custom operator '
                            'directory via the environment variable: {}'
                            .format(lib_name, CUSTOM_OPERATIONS))
        path = os.environ[CUSTOM_OPERATIONS] + "/" + lib_name
        if not os.path.isfile(path):
            raise FileNotFoundError('File not found in path: {}'.format(path))
        dyndep.InitOpsLibrary(path)
        custom_ops_loaded = True


class Caffe2ConsistentParameterServer(Optimizer):
    def __init__(self, optimizer: Optimizer, comm=CommunicationNetwork()):
        super(Caffe2ConsistentParameterServer, self).__init__(optimizer.network)
        self.communication = comm
        self.parameter_optimizer = optimizer

    def build(self):
        self.parameter_optimizer.build()
        self.build_consistent_parameter_server_gradients(self.network, self.communication)

    def build_consistent_parameter_server_gradients(self, network: Caffe2Network, comm_network: CommunicationNetwork):
        _load_custom_dll()

        gradients = network.gradient()

        ptr = comm_network.get_comm_numpy_ptr()
        network.feed_tensor("mpi_comm", ptr, device_option=core.DeviceOption(caffe2_pb2.CPU))

        # Copy GPU data to CPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                grad_name_from = grad_name + "_cpu" if network.is_cuda else grad_name
                with core.DeviceScope(network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.EnsureCPUOutput([grad_name], grad_name_from)

        # Invoke MPI
        for (param_name, grad_name) in gradients:
            grad_name_from = grad_name + "_cpu" if network.is_cuda else grad_name
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                # we use the copied tensor as input
                network.train_model.DMpiReduceMean([grad_name_from, "mpi_comm"], grad_name + "_buffer")
                network.train_model.DMpiBroadcast([grad_name + "_buffer", "mpi_comm"], grad_name_from)

        # We have to copy back the communicated tensor if we are on the GPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                with core.DeviceScope(network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.CopyFromCPUInput([grad_name + "_cpu"], grad_name)


class Caffe2ConsistentDecentralized(Optimizer):
    def __init__(self, optimizer: Optimizer, comm=CommunicationNetwork()):
        super(Caffe2ConsistentDecentralized, self).__init__(optimizer.executor)
        self.communication = comm
        self.parameter_optimizer = optimizer

    def build(self):
        self.parameter_optimizer.build()
        self.build_allreduce_gradients(self.executor.network, self.communication)

    def build_allreduce_gradients(self, network: Caffe2Network, comm_network: CommunicationNetwork):
        _load_custom_dll()

        gradients = network.gradient()

        ptr = comm_network.get_comm_numpy_ptr()
        network.feed_tensor("mpi_comm", ptr, device_option=core.DeviceOption(caffe2_pb2.CPU))

        # Copy GPU data to CPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                with core.DeviceScope(self.network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.EnsureCPUOutput([grad_name], grad_name + "_cpu")

        # Invoke MPI
        for (param_name, grad_name) in gradients:
            grad_name_from = grad_name + "_cpu" if network.is_cuda else grad_name
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                # we use the copied tensor as input
                network.train_model.DMpiAllReduceMean([grad_name_from, "mpi_comm"], grad_name_from)

        # We have to copy back the communicated tensor if we are on the GPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                with core.DeviceScope(self.network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.CopyFromCPUInput([grad_name + "_cpu"], grad_name)



class Caffe2ConsistentNeighbors(Optimizer):
    """
    Source: https://arxiv.org/pdf/1705.09056.pdf
    :return:
    """

    def __init__(self, optimizer: Optimizer, comm=CommunicationNetwork()):
        super(Caffe2ConsistentNeighbors, self).__init__(optimizer.network)
        self.communication = comm
        self.parameter_optimizer = optimizer

    def build(self):
        self.parameter_optimizer.build()
        self.build_allreduce_neighbors_gradients(self.network, self.communication)

    def build_allreduce_neighbors_gradients(self, network: Caffe2Network, comm_network: CommunicationNetwork):
        _load_custom_dll()

        gradients = network.gradient()

        ptr = comm_network.get_comm_neighbor_numpy_ptr()
        network.feed_tensor("mpi_comm", ptr, device_option=core.DeviceOption(caffe2_pb2.CPU))

        # Copy GPU data to CPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                with core.DeviceScope(self.network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.EnsureCPUOutput([grad_name], grad_name + "_cpu")

        # Invoke MPI
        for (param_name, grad_name) in gradients:
            grad_name_from = grad_name + "_cpu" if network.is_cuda else grad_name
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                # we use the copied tensor as input
                network.train_model.DMpiAllReduceMean([grad_name_from, "mpi_comm"], grad_name_from)

        # We have to copy back the communicated tensor if we are on the GPU
        if network.is_cuda:
            for (param_name, grad_name) in gradients:
                with core.DeviceScope(self.network.device_option):
                    # we copy on the same device on where mpi_comm is
                    network.train_model.CopyFromCPUInput([grad_name + "_cpu"], grad_name)
