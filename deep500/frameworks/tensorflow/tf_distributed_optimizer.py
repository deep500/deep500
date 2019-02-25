from deep500.frameworks.tensorflow.tf_optimizers import TFOptimizer
from deep500.lv3.communication import CommunicationNetwork

import tensorflow as tf

class HorovodDistributedOptimizer(TFOptimizer):
    def __init__(self, optimizer: TFOptimizer, comm=None):
        super().__init__(optimizer.executor, optimizer.loss)

        try:
            import horovod.tensorflow as hvd
        except ImportError:
            raise ImportError('Cannot import Horovod')
            
        hvd.init()
        self.op = hvd.DistributedOptimizer(optimizer.op)


        if comm is None:
            comm = CommunicationNetwork()
        self.communication = comm
        self.original_optimizer = optimizer


    def as_operator(self):
        try:
            import horovod.tensorflow as hvd
        except ImportError:
            raise ImportError('Cannot import Horovod')
        
        self.network.session_config.gpu_options.visible_device_list = str(hvd.local_rank())
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        self.network.add_hooks(hooks)
        return self.op.minimize(self.network.fetch_internal_tensor(self.loss))
        
