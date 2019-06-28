from typing import Dict, List
import numpy as np

from deep500.utils.metric import TestMetric
from deep500.lv2.dataset import Dataset
from deep500.lv2.event import RunnerEvent, SamplerEvent
from deep500.lv2.summaries import TrainingStatistics

def DefaultTrainingMetrics():
    return [TrainingAccuracy(), TestAccuracy()]

# Training metrics
class TrainingAccuracy(RunnerEvent, TestMetric):
    def __init__(self, max_acc=False):
        super().__init__()
        self.accuracy = -1.0
        self.max_acc = max_acc

    def after_training(self, runner, training_stats: TrainingStatistics):
        if len(training_stats.train_summaries) > 0:
            if self.max_acc:
                self.accuracy = max(s.accuracy for s in training_stats.train_summaries)
            else:
                self.accuracy = training_stats.train_summaries[-1].accuracy

    def measure(self, unused_inputs, unused_outputs, unused_ref) -> float:
        return self.accuracy


class TestAccuracy(RunnerEvent, TestMetric):
    def __init__(self, max_acc=False):
        super().__init__()
        self.accuracy = -1.0
        self.max_acc = max_acc

    def after_training(self, runner, training_stats: TrainingStatistics):
        if len(training_stats.test_summaries) > 0:
            if self.max_acc:
                self.accuracy = max(s.accuracy for s in training_stats.test_summaries)
            else:
                self.accuracy = training_stats.test_summaries[-1].accuracy

    def measure(self, unused_inputs, unused_outputs, unused_ref) -> float:
        return self.accuracy

class DatasetAccuracy(TestMetric):
    """ Tests classification accuracy of executor on a given dataset. """
    def __init__(self, ds: Dataset, batch_size: int = 64):
        self.ds = ds
        self.batch_size = batch_size
    
    def measure(self, runner, unused_outputs, unused_ref) -> float:
        """ Returns percentage. """
        if runner.network_output is None:
            raise ValueError('Network output must be defined for accuracy')
        if (getattr(self.ds, 'label_node', False) == False or 
                self.ds.label_node is None):
            raise ValueError('Label node must be defined in dataset')

        correct_results = 0
        for i in len(self.ds) // self.batch_size:
            out = runner.executor.inference(self.ds[i*self.batch_size:(i+1)*self.batch_size])
            output = out[runner.network_output]
            ground_truth = out[self.ds.label_node]
            correct_results += np.sum(ground_truth == np.argmax(output))

        # Remainder of dataset
        remainder = len(self.ds) % self.batch_size
        if remainder != 0:
            out = runner.executor.inference(self.ds[-remainder:])
            output = out[runner.network_output]
            ground_truth = out[self.ds.label_node]
            correct_results += np.sum(ground_truth == np.argmax(output))
        
        # Compute accuracy
        return correct_results / len(self.ds) * 100


class DatasetBias(TestMetric):
    """ Tests for bias in classification-based datasets by creating
        a histogram of sampled classes. """
    def __init__(self, num_classes: int, label_node: str, reruns: int = 1000):
        """ Creates a DatasetBias metric.
            @param num_classes Total number of classes in dataset.
            @param label_node Name of label node in graph for outputs.
            @param reruns The number of reruns for this metric.
        """
        self.hist = [0] * num_classes
        self._reruns = reruns
        self._label = label_node

    @property
    def reruns(self) -> int:
        return self._reruns

    def end(self, outputs: Dict[str, np.ndarray]):
        out = outputs[self._label]
        for outclass in np.nditer(out):
            self.hist[outclass] += 1

    def measure(self, *unused) -> List[int]:
        return self.hist

class SamplerEventMetric(SamplerEvent, TestMetric):
    """ A metric wrapper that invokes begin() and end() upon sampling.
        Example use: measuring sampler latency as part of training.
    """
    def __init__(self, metric: TestMetric):
        self.metric = metric

    def before_sampling(self, sampler, batch_size):
        self.metric.begin(batch_size)

    def after_sampling(self, sampler, resulting_batch_size):
        self.metric.end(resulting_batch_size)

    @property
    def reruns(self):
        return self.metric.reruns

    def measure(self, *args):
        return self.metric.measure(*args)

    def measure_summary(self, *args):
        return self.metric.measure_summary(*args)

