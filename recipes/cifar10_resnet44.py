""" A recipe for running the CIFAR-10 dataset with ResNet-44 and a momentum
    optimizer, with metrics for final test accuracy. """

import deep500 as d5
import deep500.frameworks.reference as d5ref
from recipes.recipe import run_recipe

# Using PyTorch as the framework
import deep500.frameworks.pytorch as d5fw


# Fixed Components
FIXED = {
    'model': 'resnet',
    'model_kwargs': dict(depth=44),
    'dataset': 'cifar10',
    'epochs': 90
}

# Mutable Components
MUTABLE = {
    'batch_size': 64,
    'executor': d5fw.from_model,
    'executor_kwargs': dict(device=d5.GPUDevice()),
    'train_sampler': d5.ShuffleSampler,
    'train_sampler_kwargs': dict(transformations=[
        d5ref.Crop('0', 'label', (32, 32),
                   random_crop=True, padding=(4, 4)),
        d5ref.RandomFlip('0', 'label'),
        d5ref.Cutout('0', 'label', 1, 16)
    ]),
    'optimizer': d5fw.MomentumOptimizer,
    'optimizer_args': (0.1, 0.9),
    'optimizer_kwargs': dict(weight_decay=1e-4),
    'events': [d5.training_events.EpochHPSchedule(lr=[(0, 1e-1),
                                                      (81, 1e-2),
                                                      (122, 1e-3),
                                                      (164, 1e-4)]),
               d5.training_events.TerminalBarEvent()]
}

# Acceptable Metrics
METRICS = [
    (d5.TestAccuracy(), 93.0)
]


if __name__ == '__main__':
    run_recipe(FIXED, MUTABLE, METRICS) or exit(1)
