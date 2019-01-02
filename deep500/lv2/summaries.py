from typing import List

import numpy as np

import time


class EpochSummary(object):
    def __init__(self, is_train, epoch):
        self.losses = []
        self.current_loss = -1
        self.loss_index = -1
        self.is_train = is_train
        self.wrong = None
        self.wrong_batch = []
        self.accuracy = -1
        self.n_batches = -1
        self.epoch = epoch
        self.avg_time_inference = 0
        self.n_steps_inference = 0
        self.start_time = time.time()
        self.time_used = -1
        self.avg_loss = 0

        self.time_used_inference = []
        self.time_used_optimizing = []

    def to_dict(self):
        table = dict()
        table['epoch'] = self.epoch
        table['avg_loss'] = self.avg_loss
        table['accuracy'] = self.accuracy
        table['wrong'] = self.wrong
        table['n_batches'] = self.n_batches
        table['avg_time_inference'] = self.avg_time_inference
        table['train'] = self.is_train
        table['start_time'] = self.start_time
        table['time_used'] = self.time_used
        return table

class TrainingStatistics(object):
    def __init__(self, n_batch_size_train: int, n_batch_size_test: int):
        self.n_batch_size_train = n_batch_size_train
        self.n_batch_size_test = n_batch_size_test
        self.clear()

    def clear(self):
        self.n_batches_used = 0
        self.test_summaries = [] # type: List[EpochSummary]
        self.train_summaries = [] # type: List[EpochSummary]
        self.current_summary = None # type: EpochSummary
        self.current_epoch = 0

    def _to_dict(self, summaries, batch_size):
        times_used = np.cumsum(np.array([s.time_used for s in summaries]))
        for i, summary in enumerate(summaries):
            dictly = summary.to_dict()

            dictly['total_used_time'] = times_used[i]
            dictly['train_batch_size'] = batch_size

            df = df.append(dictly, ignore_index=True)
        return df
        
    def to_dict(self):
        df_train = self._to_dict(self.train_summaries, self.n_batch_size_train)
        df_test = self._to_dict(self.test_summaries, self.n_batch_size_test)
        return df_train, df_test
