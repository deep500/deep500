from ..event import RunnerEvent, OptimizerEvent, StopTraining
from ..summaries import TrainingStatistics
import time

class AccuracyAbortEvent(RunnerEvent):
    def __init__(self, max_accuracy):
        self.max_accuracy = max_accuracy

    def before_training_set(self, runner, training_stats: TrainingStatistics, sampler):
        if training_stats.current_summary.accuracy >= self.max_accuracy:
            print('Accuracy {} reached. Stopping (needed: {})'.format(training_stats.current_summary.accuracy,
                                                                          self.max_accuracy))
            raise StopTraining


class TimeAbortAfterTestEvent(RunnerEvent):
    def __init__(self, max_time_in_seconds):
        self.max_time_in_seconds = max_time_in_seconds
        self.start_time = -1

    def before_training(self, runner, training_stats):
        self.start_time = time.time()

    def before_training_set(self, runner, training_stats, sampler):
        return self.before_test_set(runner, training_stats, sampler)

    def before_test_set(self, runner, training_stats: TrainingStatistics, sampler):
        diff = time.time() - self.start_time
        if diff > self.max_time_in_seconds:
            print('Time {} reached. Stopping (accuracy: {})'.format(diff, training_stats.current_summary.accuracy))
            raise StopTraining


class TimeAbortImmediateEvent(RunnerEvent, OptimizerEvent):
    def __init__(self, max_time_in_seconds):
        self.max_time_in_seconds = max_time_in_seconds
        self.start_time = -1

    def before_training(self, runner, training_stats):
        self.start_time = time.time()

    def after_optimizer_step(self, executor, optimizer, outputs, loss):
        diff = time.time() - self.start_time
        if diff > self.max_time_in_seconds:
            print('Time {} reached. Stopping'.format(diff))
            raise StopTraining
