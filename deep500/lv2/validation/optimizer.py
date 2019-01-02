from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np

import deep500 as d5


def DefaultOptimizerMetrics():
    return [d5.L2Error(), d5.WallclockTime()]

class OneInputDataset(d5.Dataset):
    def __init__(self, input, label):
        super().__init__(events)
        self.input = input
        if label is not None:
            self.label = label
        
    def __getitem__(self, index):
        if self.label is None:
            return self.input

        result = {}
        result.update(self.input)
        result.update(self.label)
        return result

    def __len__(self):
        return 1

def test_optimizer(optimizer: d5.Optimizer, 
                   input: Union[d5.Sampler, d5.Dataset, Dict[str, np.ndarray],
                                Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]],
                   expected_params: List[np.ndarray], 
                   metrics=DefaultOptimizerMetrics(), 
                   events: List[d5.OptimizerEvent] = []) -> List[Any]:
    """ Measures an optimizer by taking one step, invoking events,
        and running the metrics.
        @param optimizer The optimizer to test.
        @param iterations Number of iterations to run.
        @param input The inputs to feed. Can either be a Sampler, a Dataset 
                     (which will use a ShuffleSampler), an unsupervised
                     input (as a dictionary of node names to values), or a 
                     tuple of two dictionaries (input, label).
        @param expected_params The reference parameters after an optimization 
                               step.
        @param metrics The metrics to measure.
        @param events Events to invoke (can overlap with metric objects).
        @return List of metric outputs.
    """
    # Create Sampler according to input
    if isinstance(input, d5.Sampler):
        sampler = input
    elif isinstance(input, d5.Dataset):
        sampler = d5.ShuffleSampler(input, 1)
    elif isinstance(input, dict):
        sampler = d5.ShuffleSampler(OneInputDataset(input, None), 1)
    elif (isinstance(input, tuple) and len(input) == 2 and 
                isinstance(input[0], dict) and isinstance(input[1], dict)):
        sampler = d5.ShuffleSampler(OneInputDataset(input[0], input[1]), 1)
    else:
        raise ValueError('Invalid input type')

    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for metric in normal_metrics:
        metric.begin(input)
    outputs = optimizer.train(1, sampler, events)
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            metric.begin(input)
            outputs = optimizer.train(1, sampler, events)
            metric.end(outputs)

    # Execute metrics
    results = []
    for metric in metrics:
        result = metric.measure(input, outputs, expected_params)
        results.append(result)
        summary = metric.measure_summary(inputs, outputs, expected_params)
        print("{}: {}".format(
                type(metric).__name__, summary))

    return results
