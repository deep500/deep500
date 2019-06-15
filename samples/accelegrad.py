import numpy as np
import deep500 as d5

# Sample that defines a new optimizer and trains a network
class AcceleGradOptimizer(d5.ThreeStepOptimizer):
    """ The AcceleGrad optimizer, as originally defined in:
        Kfir Y. Levy, Alp Yurtsever, Volkan Cevher,
        "Online Adaptive Methods, Universality and Acceleration," in NeurIPS'18.
        https://arxiv.org/abs/1809.02864
    """
    def __init__(self, executor: d5.GraphExecutor, lr=0.999, G=0, D=1.0, eps=1e-6):
        super().__init__(executor)
        self.lr = lr
        self.eps = eps
        self.squares = {}
        self.z = {}
        self.y = {}
        self.t = -1

        self.G = G
        self.D = D

        self.alpha_t = 0
        self.tau_t = 0
        self.x_t = 0

        self.init = False

    def new_input(self):
        self.t = self.t + 1
        self.alpha_t = 1 if 0 <= self.t <= 2 else 1 / 4 * (self.t + 1)
        self.tau_t = 1 / self.alpha_t

    def prepare_param(self, param_name):
        param = self.executor.network.fetch_tensors([param_name])[0]
        if not self.init:
            self.y[param_name] = param
            self.z[param_name] = param
            self.squares[param_name] = 0
        y = self.y[param_name]
        z = self.z[param_name]
        new_param = self.tau_t * z + (1 - self.tau_t) * y
        self.executor.network.feed_tensor(param_name, new_param)

    def update_rule(self, grad, old_param, param_name):
        squared_grad = self.squares[param_name]
        squared_grad += self.alpha_t ** 2 * np.linalg.norm(grad) ** 2

        eta_t = 2 * self.D / np.sqrt(self.G ** 2 + squared_grad + self.eps)
        z_t = self.z[param_name]

        z_t2 = z_t - self.alpha_t * eta_t * grad
        y_t2 = old_param - eta_t * grad

        self.z[param_name] = z_t2
        self.y[param_name] = y_t2
        self.squares[param_name] = squared_grad
        adjusted_lr = self.lr / (self.eps + np.sqrt(squared_grad))

        self.init = False
        return old_param - adjusted_lr * grad

if __name__ == '__main__':
    from deep500 import networks as d5net, datasets as d5ds
    from deep500.frameworks import tensorflow as d5tf
    from deep500.frameworks import reference as d5ref
    batch_size = 1024

    # Create network and dataset
    net, innode, outnode = d5net.create_model('simple_cnn', batch_size)
    net.add_operation(d5.ops.SoftmaxCrossEntropy([outnode, 'label'], 'loss'))
    train, test = d5ds.load_dataset('mnist', innode, 'label')

    # Create executor and optimizer
    executor = d5tf.from_model(net)
    opt = AcceleGradOptimizer(executor)
    
    # Run training
    d5.test_training(executor, train, test, opt, 5, batch_size, outnode)
