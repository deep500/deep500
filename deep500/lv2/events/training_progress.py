from tqdm.auto import tqdm


from ..event import RunnerEvent, OptimizerEvent
from ..summaries import TrainingStatistics


class TerminalBarEvent(RunnerEvent, OptimizerEvent):
    def __init__(self, total_epochs=-1):
        self.bar = None
        self._epoch = 1
        self._total_epochs = total_epochs
    
    def before_training(self, runner, training_stats: TrainingStatistics):
        self.stats = training_stats
        self.runner = runner

    def after_epoch(self, epoch, runner, training_stats):
        self._epoch += 1

    def before_training_set(self, runner, training_stats, sampler):        
        # Compute total number of batches according to sampler and dataset
        num_batches = len(sampler)
        if num_batches == 0:
            self.bar = tqdm()
        else:
            self.bar = tqdm(range(num_batches))

        if self._total_epochs < 0:
            self.bar.set_description('Training (epoch %d)' % self._epoch)
        else:
            self.bar.set_description('Training (epoch %d/%d)' % (self._epoch, self._total_epochs))

    def after_optimizer_step(self, executor, optimizer, outputs, loss):
        if self.stats.current_summary.wrong is not None:
            bs = self.runner.train_set.batch_size
            acc = (bs - self.stats.current_summary.wrong_batch[-1]) / bs
            self.bar.set_postfix(loss_avg=self.stats.current_summary.avg_loss,
                                 batch_acc=acc*100)
        else:
            self.bar.set_postfix(loss_avg=self.stats.current_summary.avg_loss)
        self.bar.update(1)

    def after_training_set(self, runner, training_stats):
        self.bar.close()

    def before_test_set(self, runner, training_stats, sampler):
        # Compute total number of batches according to sampler and dataset
        num_batches = len(sampler)
        if num_batches == 0:
            self.bar = tqdm()
        else:
            self.bar = tqdm(range(num_batches))

        self.bar.set_description('Testing')

    def after_test_batch(self, runner, training_stats: TrainingStatistics, output):
        if training_stats.current_summary.wrong is not None:
            seen_samples = training_stats.current_summary.n_steps_inference * runner.test_set.batch_size
            acc = (seen_samples - training_stats.current_summary.wrong) / seen_samples
            self.bar.set_postfix(test_loss=training_stats.current_summary.avg_loss,
                                 accuracy=acc*100)
        else:
            self.bar.set_postfix(test_loss=training_stats.current_summary.avg_loss)
        self.bar.update(1)

    def after_test_set(self, runner, training_stats: TrainingStatistics):
        self.bar.close()
