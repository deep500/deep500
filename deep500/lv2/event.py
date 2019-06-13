from deep500 import Event
from deep500.lv2.summaries import TrainingStatistics

class StopTraining(Exception):
    pass

class TrainingEvent(Event):
    """ Any event that relates to Deep500's Level 2. """
    pass
    
class OptimizerEvent(TrainingEvent):
    """ Optimizer events. """
    
    def before_optimizer_step(self, executor, optimizer, inputs):
        pass

    def after_optimizer_step(self, executor, optimizer, outputs, loss):
        pass

class SamplerEvent(TrainingEvent):
    """ Data staging and sampling events. """
        
    def on_sampler_reset(self, sampler):
        pass
    
    def before_sampling(self, sampler, minibatch_size):
        pass
        
    def after_sampling(self, sampler, samples):
        pass
    
class RunnerEvent(TrainingEvent):
    """ Training/test set loop events. """

    def before_training(self, runner, training_stats: TrainingStatistics):
        pass
    
    def after_training(self, runner, training_stats: TrainingStatistics):
        pass

    def before_training_set(self, runner, training_stats: TrainingStatistics, 
                            sampler):
        pass
    
    def after_training_set(self, runner, training_stats: TrainingStatistics):
        pass

    def before_test_set(self, runner, training_stats: TrainingStatistics, 
                        sampler):
        pass

    def after_test_set(self, runner, training_stats: TrainingStatistics):
        pass

    def before_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        pass
        
    def after_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        pass
    
    def before_test_batch(self, runner, training_stats: TrainingStatistics, input):
        pass

    def after_test_batch(self, runner, training_stats: TrainingStatistics, output):
        pass
