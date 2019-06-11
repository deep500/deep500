""" A recipe for running the CIFAR-10 dataset with ResNet-44 and a momentum
    optimizer, with metrics for final test accuracy. """

import deep500 as d5
from recipes.recipe import run_recipe

# Using PyTorch as the framework
import deep500.frameworks.pytorch as d5fw


# Fixed Components
FIXED = {
    'model': 'resnet',
    'model_kwargs': dict(depth=44),
    'dataset': 'cifar10',
    'train_sampler': d5.ShuffleSampler,
    'epochs': 1
}

# Mutable Components
MUTABLE = {
    'batch_size': 64,
    'executor': d5fw.from_model,
    'executor_kwargs': dict(device=d5.GPUDevice()),
    'optimizer': d5fw.MomentumOptimizer,
    'optimizer_args': (0.1, 0.9),
}

# Acceptable Metrics
METRICS = [
    (d5.TestAccuracy(), 93.0)
]


if __name__ == '__main__':
    run_recipe(FIXED, MUTABLE, METRICS) or exit(1)
