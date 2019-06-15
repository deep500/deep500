import os

from ..event import RunnerEvent
from ..summaries import TrainingStatistics



class BoxPlotGeneratorEvent(RunnerEvent):
    def __init__(self, file_name="box_plot", folder=os.getcwd()):
        self.path = folder + '/' + file_name

    def after_training(self, runner, training_stats: TrainingStatistics):
        import matplotlib.pyplot as plt

        if len(training_stats.train_summaries) > 0 and (len(training_stats.train_summaries[0].time_used_inference) == 0
                and len(training_stats.train_summaries[0].time_used_optimizing) == 0):
            raise ValueError('To generate box-plots, please train with the '
                             'Trainer object with collect_all_times=True')
        
        inference_test_data = [s.time_used_inference for s in training_stats.test_summaries]
        optimizing_time_train = [s.time_used_optimizing for s in training_stats.train_summaries]

        plt.figure()
        plt.title('Time used for inference')
        plt.xlabel('Epoch')
        plt.ylabel('time used')
        plt.boxplot(inference_test_data, 1, '')
        plt.savefig(self.path + "_inference_test")
        print('Box plot written to: {}.png'.format(self.path + "_inference_test"))
        plt.close()

        plt.figure()
        plt.title('Time used for optimization (inference + gradient update)')
        plt.xlabel('Epoch')
        plt.ylabel('time used')
        plt.boxplot(optimizing_time_train, 1, '')
        plt.savefig(self.path + "_optimizing_train")
        print('Box plot written to: {}.png'.format(self.path + "_optimizing_train"))
        plt.close()
