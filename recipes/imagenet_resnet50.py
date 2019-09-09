""" A recipe for running the ImageNet (ILSVRC2012, 1000 classes) dataset 
    with ResNet-50 and a momentum optimizer, with metrics for final test 
    accuracy. """

import deep500 as d5


# Fixed Components
FIXED = {
    'model': 'resnet',
    'model_kwargs': dict(depth=50),
    'dataset': 'imagenet',
    'dataset_kwargs': dict(batch_size=128),
    'epochs': 90
}


def weight_decay(name: str, param, grad):
    if 'bias' in name or len(param.shape) == 1:
        return grad
    grad += 1e-4 * param
    return grad


import deep500.frameworks.pytorch as d5fw

# Mutable Components
MUTABLE = {
    'batch_size': 128,
    'executor': d5fw.from_model,
    'executor_kwargs': dict(device=d5.GPUDevice()),
    'optimizer': d5fw.MomentumOptimizer,
    'optimizer_args': (0.1, 0.9),
    'optimizer_kwargs': dict(gradient_modifier=weight_decay),
    'events': [d5.training_events.EpochHPSchedule(
                    lr=[(0, 1e-1), (30, 1e-2), (60, 1e-3), (80, 1e-4)]),
               d5.training_events.TerminalBarEvent()]
}

# Acceptable Metrics
METRICS = [
    (d5.TestAccuracy(max_acc=True), 75.0)
]


if __name__ == '__main__':
    d5.run_recipe(FIXED, MUTABLE, METRICS) or exit(1)
