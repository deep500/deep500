# Measures the time it takes to load a minibatch from a dataset.
# NOTE: ImageNet uses the TFRecord dataset format and requires TensorFlow.
import deep500 as d5
from deep500 import datasets as d5ds
import scipy.misc
import time

BATCH_SIZE = 128

def test_dataset(dataset_name):
    print('\n=========================')
    print('Loading dataset', dataset_name)
    sampler, _ = d5ds.load_dataset(dataset_name, 'in', 'label', 
                                   batch_size=BATCH_SIZE)
    print('Dataset loaded')
    # ImageNet already outputs a sampler due to the file format
    if dataset_name != 'imagenet':
        sampler = d5.ShuffleSampler(sampler, BATCH_SIZE)
    
    # In this case, a simple WallclockTime metric would also suffice,
    # but a SamplerEventMetric can also be used in the context of training.
    metrics = [d5.SamplerEventMetric(d5.WallclockTime())]
    d5.test_sampler(sampler, metrics=metrics)


if __name__ == '__main__':
    # Test all datasets
    for ds in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imagenet']:
        try:
            test_dataset(ds)
        except ImportError as ex:
            print('Dependency error, skipping dataset %s: %s' % (ds, str(ex)))
