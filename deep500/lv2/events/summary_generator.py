import numpy as np

from deep500.lv1.event import ExecutorEvent
from deep500.lv2.event import RunnerEvent, OptimizerEvent
from ..summaries import TrainingStatistics, EpochSummary

import time

class SummaryGeneratorEvent(RunnerEvent, OptimizerEvent, ExecutorEvent):
    """
    Training statistics generator.
    """

    def __init__(self, training_stats: TrainingStatistics):
        super().__init__()
        training_stats.test_summaries.clear()
        training_stats.train_summaries.clear()
        self.training_stats = training_stats

    def before_training_set(self, runner, training_stats: TrainingStatistics, sampler):
        self.runner = runner
        training_stats.current_summary = EpochSummary(True, training_stats.current_epoch)
        training_stats.current_summary.n_batches = len(runner.train_set) // runner.train_set.batch_size

    def after_training_set(self, runner, training_stats: TrainingStatistics):
        training_stats.current_summary.time_used = time.time() - training_stats.current_summary.start_time

        if (runner.train_set.dataset and runner.network_output and 
                training_stats.current_summary.wrong is not None):
            ds_len = len(runner.train_set) * runner.train_set.batch_size
            training_stats.current_summary.accuracy = (
                100 / ds_len * (ds_len - training_stats.current_summary.wrong))

        training_stats.train_summaries.append(training_stats.current_summary)

    def before_optimizer_step(self, executor, optimizer, input):
        self.inp = input

    def after_optimizer_step(self, executor, optimizer, output, loss):
        training_stats = self.training_stats
        training_stats.current_summary.losses.append(loss)
        n = len(training_stats.current_summary.losses) + 1
        training_stats.current_summary.avg_loss = ((n - 1) / n) * training_stats.current_summary.avg_loss + (1 / n) * loss
        training_stats.current_summary.current_loss = loss
        training_stats.current_summary.loss_index += 1
        training_stats.n_batches_used += 1

        # Accuracy for classification
        if (self.inp and self.runner.train_set.dataset
                and self.runner.network_output and self.runner.network_output in output):
                y_corr = self.inp[self.runner.train_set.dataset.label_node]
                y_network = output[self.runner.network_output]

                if training_stats.current_summary.wrong is None:
                    training_stats.current_summary.wrong = 0
                wrong = np.sum(y_corr != np.argmax(y_network, axis=1))
                training_stats.current_summary.wrong += wrong
                training_stats.current_summary.wrong_batch.append(wrong)

    def before_test_set(self, runner, training_stats: TrainingStatistics, sampler):
        self.runner = runner
        training_stats.current_summary = EpochSummary(False, training_stats.current_epoch)
        training_stats.current_summary.n_batches = len(runner.test_set) // runner.test_set.batch_size

    def before_executor(self, input):
        self.inp = input
        self.time_before_inference = time.time()

    def after_inference(self, output):
        time_after_inference = time.time()
        training_stats = self.training_stats
        runner = self.runner

        training_stats.current_summary.n_steps_inference += 1
        avg = training_stats.current_summary.avg_time_inference
        used_time = time_after_inference - self.time_before_inference

        if avg == 0:
            training_stats.current_summary.avg_time_inference = used_time
        else:
            n = training_stats.current_summary.n_steps_inference
            training_stats.current_summary.avg_time_inference = ((n - 1) / n) * avg + (1 / n) * used_time

        # Accuracy for classification
        if self.inp and runner.test_set.dataset and runner.network_output:
            y_corr = self.inp[runner.test_set.dataset.label_node]
            y_network = output[runner.network_output]

            if training_stats.current_summary.wrong is None:
                training_stats.current_summary.wrong = 0
            training_stats.current_summary.wrong += np.sum(y_corr != np.argmax(y_network, axis=1))
        # Loss
        loss = output[runner.optimizer.loss]
        training_stats.current_summary.losses.append(loss)
        n = len(training_stats.current_summary.losses) + 1
        training_stats.current_summary.avg_loss = ((n - 1) / n) * training_stats.current_summary.avg_loss + (1 / n) * loss
        training_stats.current_summary.current_loss = loss

    def after_test_set(self, runner, training_stats: TrainingStatistics):
        training_stats.current_summary.time_used = time.time() - training_stats.current_summary.start_time
        
        if runner.test_set.dataset and runner.network_output:
            ds_len = len(runner.test_set) * runner.test_set.batch_size
            training_stats.current_summary.accuracy = (
                100 / ds_len * (ds_len - training_stats.current_summary.wrong))

        training_stats.test_summaries.append(training_stats.current_summary)

    def after_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        training_stats.current_epoch += 1


class SummaryGeneratorInferenceEvent(SummaryGeneratorEvent):
    """
    This event extends the statistics generator by adding inference and optimizer time.
    """
    def __init__(self, training_stats: TrainingStatistics):
        super().__init__(training_stats)

    def after_inference(self, output):
        curtime = time.time()
        super().after_inference(output)
        used_time = curtime - self.time_before_inference
        self.training_stats.current_summary.time_used_inference.append(used_time)

    def before_optimizer_step(self, executor, optimizer, inputs):
        super().before_optimizer_step(executor, optimizer, inputs)
        self.time_before_optimizing = time.time()

    def after_optimizer_step(self, executor, optimizer, outputs, loss):
        curtime = time.time()
        super().after_optimizer_step(executor, optimizer, outputs, loss)
        self.training_stats.current_summary.time_used_optimizing.append(curtime - self.time_before_optimizing)
