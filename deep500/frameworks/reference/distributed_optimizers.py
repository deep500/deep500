import abc

from deep500.lv2.optimizer import ThreeStepOptimizer
from deep500.lv3.distributed_optimizer import DistributedOptimizer
from deep500.lv3.communication import CommunicationNetwork

import numpy as np


class ConsistentParameterServer(DistributedOptimizer):

    def step(self, inputs):
        self.base_optimizer.new_input()
        [self.base_optimizer.prepare_param(param) for param in self.network.get_params()]
        outputs = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)

        for (param_name, grad_name) in self.network.gradient(self.base_optimizer.loss):
            param, grad = self.network.fetch_tensors([param_name, grad_name])

            grad = self.communication.gather_at_root(grad)

            if self.communication.rank == 0:
                grad = np.mean(grad, axis=0)
                param = self.base_optimizer.update_rule(grad, param, param_name)

            self.communication.sync_all_with_root(param)

            self.network.feed_tensor(param_name, param)

        return outputs


class InconsistentParameterServer(DistributedOptimizer):

    def step(self, inputs):
        if self.communication.rank != 0:
            self.base_optimizer.new_input()
            for param in self.network.get_params():
                self.base_optimizer.prepare_param(param)
            output = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)
            for k, (param_name, grad_name) in enumerate(self.network.gradient(self.base_optimizer.loss)):
                param, grad = self.network.fetch_tensors([param_name, grad_name])
                self.communication.send_to_root(grad, tag=k)
                param = self.communication.wait_for_root()
                self.network.feed_tensor(param_name, param)

        # we only loop over the gradients so that we know when we can stop
        # else we would have a infinity loop with a message to abort sometime
        # for the moment we can do it this way
        if self.communication.rank == 0:
            # we repeat this here for every worker once
            for z in range(self.communication.size - 1):
                self.base_optimizer.new_input()
                for param in self.network.get_params():
                    self.base_optimizer.prepare_param(param)
                output = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)
                gradients = self.network.gradient(self.base_optimizer.loss)
                for k, (param_name, grad_name) in enumerate(gradients):
                    grad, source, tag = self.communication.wait_for_any_rank()
                    param_name = gradients[tag][0]
                    param = self.network.fetch_tensors([param_name])[0]
                    param = self.base_optimizer.update_rule(grad, param, param_name)
                    self.communication.send_to_rank(param, source, tag)
                    self.network.feed_tensor(param_name, param)
        return output


class ConsistentDecentralized(DistributedOptimizer):

    def step(self, inputs):
        #batch_size = list(inputs.values())[0].shape[0]
        self.base_optimizer.new_input()
        for param in self.network.get_params():
            self.base_optimizer.prepare_param(param)
        output = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)
        gradients = self.network.gradient(self.base_optimizer.loss)
        for param_name, grad_name in gradients:
            param, grad = self.network.fetch_tensors([param_name, grad_name])
            grad = self.communication.sync_all(grad) / self.communication.size
            param = self.base_optimizer.update_rule(grad, param, param_name)
            self.network.feed_tensor(param_name, param)

        return output

class ModelAverageDecentralized(DistributedOptimizer):

    def step(self, inputs):
        #batch_size = list(inputs.values())[0].shape[0]
        self.base_optimizer.new_input()
        for param in self.network.get_params():
            self.base_optimizer.prepare_param(param)
        output = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)
        gradients = self.network.gradient(self.base_optimizer.loss)
        for param_name, grad_name in gradients:
            param, grad = self.network.fetch_tensors([param_name, grad_name])
            param = self.base_optimizer.update_rule(grad, param, param_name)
            param = self.communication.sync_all(param) / self.communication.size
            self.network.feed_tensor(param_name, param)

        return output


class ConsistentNeighbors(DistributedOptimizer):
    # Follows communication scheme from https://arxiv.org/pdf/1705.09056.pdf

    def step(self, inputs):
        self.base_optimizer.new_input()
        for param in self.network.get_params():
            self.base_optimizer.prepare_param(param)
        output = self.executor.inference_and_backprop(inputs, self.base_optimizer.loss)
        gradients = self.network.gradient(self.base_optimizer.loss)
        for param_name, grad_name in gradients:
            param, grad = self.network.fetch_tensors([param_name, grad_name])
            grad = self.communication.reduce_from_neighbors(grad) / 3
            param = self.base_optimizer.update_rule(grad, param, param_name)
            self.network.feed_tensor(param_name, param)
        return output
