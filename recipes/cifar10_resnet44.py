""" A recipe for running the CIFAR-10 dataset with ResNet-44 and a momentum
    optimizer, with metrics for final test accuracy. """

import deep500 as d5
import deep500.frameworks.reference as d5ref

# Using PyTorch as the framework
import deep500.frameworks.pytorch as d5fw


# Fixed Components
FIXED = {
    'model': 'resnet',
    'model_kwargs': dict(depth=44),
    'dataset': 'cifar10',
    'epochs': 90
}


def weight_decay(name: str, param, grad):
    if 'bias' in name or len(param.shape) == 1:
        return grad
    grad += 1e-4 * param
    return grad


# Mutable Components
MUTABLE = {
    'batch_size': 64,
    'executor': d5fw.from_model,
    'executor_kwargs': dict(device=d5.GPUDevice()),
    'train_sampler': d5.ShuffleSampler,
    'train_sampler_kwargs': dict(transformations=[
        d5ref.Crop((32, 32), random_crop=True, padding=(4, 4)),
        d5ref.RandomFlip(),
        d5ref.Cutout(1, 16),
    ]),
    'optimizer': d5fw.MomentumOptimizer,
    'optimizer_args': (0.1, 0.9),
    'optimizer_kwargs': dict(gradient_modifier=weight_decay),
    'events': [d5.training_events.EpochHPSchedule(
                    lr=[(0, 1e-1), (81, 1e-2), (122, 1e-3), (164, 1e-4)]),
               d5.training_events.TerminalBarEvent()]
}

# Acceptable Metrics
METRICS = [
    (d5.TestAccuracy(max_acc=True), 93.0)
]


if __name__ == '__main__':
    d5.run_recipe(FIXED, MUTABLE, METRICS) or exit(1)
