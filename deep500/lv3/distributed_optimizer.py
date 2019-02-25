from deep500.lv2.optimizer import FirstOrderOptimizer, ThreeStepOptimizer
from deep500.lv3.communication import CommunicationNetwork

class DistributedOptimizer(FirstOrderOptimizer):
    def __init__(self, base_optimizer: ThreeStepOptimizer, comm=None):
        super().__init__(base_optimizer.executor)
        if comm is None:
            comm = CommunicationNetwork()
        self.communication = comm
        self.base_optimizer = base_optimizer
        self.loss = base_optimizer.loss

