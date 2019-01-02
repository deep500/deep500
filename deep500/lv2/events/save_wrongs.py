import os

import numpy as np

from ..event import RunnerEvent
from ..summaries import TrainingStatistics


class SaveWrongsEvent(RunnerEvent):
    def __init__(self):
        self.counter = 10
        self.curr_epoch = 0
        self.max_wrongs = 0

    def before_test_set(self, runner, training_stats, test_set):
        if runner.network_output is None:
            raise ValueError('Network output cannot be None for SaveWrongs event')

    def before_test_batch(self, runner, training_stats, input):
        self.inp = input

    def after_test_batch(self, runner, training_stats: TrainingStatistics, output):
        import matplotlib.pyplot as plt

        y_corr = self.inp[runner.test_set.dataset.label_node]
        y_network = np.argmax(output[runner.network_output], axis=1)

        x_input = inp[runner.test_set.input_node]
        wrongs = y_corr != y_network

        # x wrongs our input, y_wrong the wrong label
        (x_wrongs, y_wrong) = x_input[wrongs], y_network[wrongs]

        if len(y_wrong) > self.max_wrongs:
            self.max_wrongs = len(y_wrong)
            fig, axis = plt.subplots(1, len(y_wrong))
            for i in range(len(y_wrong)):
                # if we have only 1 element in axis we get error if we try to index it
                curr_axis = axis if len(y_wrong) == 1 else axis[i]
                curr_axis.set_title('as: {}'.format(y_wrong[i]))
                curr_axis.imshow(x_wrongs[i][0])

            file_name = 'wrong_labels_batch_epoch_{}'.format(self.curr_epoch)
            plt.savefig(file_name)
            print('Wrong labels written to: {}.png'.format(os.getcwd() + '/' + file_name))
            plt.close(fig)

    def after_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        self.curr_epoch += 1
        self.max_wrongs = 0
