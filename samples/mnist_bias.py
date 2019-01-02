import numpy as np
from tqdm import trange

import deep500 as d5
from deep500 import datasets as d5ds

# Tests for dataset sample bias using a reference ShuffleSampler object and
# a histogram of obtained classes

BATCH_SIZE = 10

if __name__ == '__main__':
    train_set, test_set = d5ds.load_mnist('input', 'label')
    tsampler = d5.ShuffleSampler(train_set, BATCH_SIZE)
    vsampler = d5.ShuffleSampler(test_set, BATCH_SIZE)
    
    [thist] = d5.test_sampler(tsampler, metrics=[d5.DatasetBias(10, 'label')])
    print('Observed training set size:', sum(thist))
    [vhist] = d5.test_sampler(vsampler, metrics=[d5.DatasetBias(10, 'label')])
    print('Observed test set size:', sum(vhist))