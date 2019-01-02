import functools
import numpy as np
import operator
import os

from ..event import RunnerEvent
from ..summaries import TrainingStatistics



class LossPlotGeneratorEvent(RunnerEvent):
    def __init__(self, file_name="loss_decrease", folder:str = os.getcwd()):
        self.path = folder + '/' + file_name

    def after_training(self, runner, training_stats: TrainingStatistics):
        import matplotlib.pyplot as plt

        losses = map(lambda s: s.losses, training_stats.train_summaries)
        losses = functools.reduce(operator.concat, losses)
        plt.figure()
        plt.plot(np.arange(0, len(losses)), losses)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')

        batch_indices = []
        for summary in training_stats.train_summaries:
            if len(batch_indices) == 0:
                batch_indices.append(summary.n_batches)
            else:
                batch_indices.append(batch_indices[-1] + summary.n_batches)

        for epoch_index in batch_indices:
            plt.axvline(epoch_index, linestyle='dashed', color='black')
        plt.savefig(self.path)
        print('Loss plot written to: {}.png'.format(self.path))
        print('Average inference time: {}'.format(training_stats.current_summary.avg_time_inference))


