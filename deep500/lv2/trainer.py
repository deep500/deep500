import copy
import os
from typing import List, Optional, Type

from deep500.lv1.graph_executor import GraphExecutor
from deep500.lv2.summaries import TrainingStatistics
from deep500.lv2.events.summary_generator import SummaryGeneratorEvent
from deep500.lv2.optimizer import Optimizer
from deep500.lv2.sampler import Sampler
from deep500.lv1.event import ExecutorEvent
from deep500.lv2.event import (TrainingEvent, OptimizerEvent, SamplerEvent, 
                               RunnerEvent, StopTraining)

from deep500.lv2.events import (SummaryGeneratorEvent, 
    SummaryGeneratorInferenceEvent, TerminalBarEvent)


def DefaultTrainerEvents(epochs):
    return [TerminalBarEvent(epochs)]


class Trainer(object):
    """ A training manager class that runs a training/test loop, with epochs 
        and invoking corresponding events. """
    def __init__(self,
                 training_sampler: Sampler,
                 validation_sampler: Optional[Sampler],
                 executor: GraphExecutor,
                 optimizer: Optimizer,
                 network_output: Optional[str] = None):
        """ Creates a Trainer object.
            @param training_sampler Sampler that samples training dataset.
            @param validation_sampler Sampler that samples validation dataset.
                                      Can be None.
            @param executor Graph executor to run the network.
            @param optimizer The optimizer to use for training.
            @param network_output The node name of the network prediction 
                                  (value or classification) for accuracy
                                  computation. Can be None.
        """
        self.train_set = training_sampler
        self.test_set = validation_sampler
        self.executor = executor
        self.optimizer = optimizer
        self.network_output = network_output

    def _train(self, stats, events, optimizer_events):
        self.train_set.reset()

        for event in events: event.before_training_set(self, stats, self.train_set)
        output = self.optimizer.train(len(self.train_set), self.train_set, 
                                      optimizer_events)
        for event in events: event.after_training_set(self, stats)

    def _test_accuracy(self, stats, events):
        if self.test_set is None: 
            return

        self.test_set.reset()

        for event in events: event.before_test_set(self, stats, self.test_set)
        for j, inp in enumerate(self.test_set):
            for event in events: event.before_test_batch(self, stats, inp)
            out = self.executor.inference(inp)
            for event in events: event.after_test_batch(self, stats, out)
        for event in events: event.after_test_set(self, stats)

    def run_loop(self, epochs,
                 events: List[TrainingEvent] = None,
                 collect_all_times: bool = False) -> TrainingStatistics:
        """
        Runs train and test set alternately for a given number of epochs.
        @param epochs Number of epochs to run the loop for.
        @param events A list of events to use in training/testing. 
                      Instances of RunnerEvent invoke the runner events,
                      instances of OptimizerEvent and SamplerEvent are also
                      invoked in the optimizer and sampler objects.
        @param collect_all_times Training statistics collect every latency
                                 of optimizer and executor steps.
        @return Training statistics for all epochs.
        """
        # Create statistics object
        stats = TrainingStatistics(self.train_set.batch_size, 
                                   (0 if self.test_set is None else 
                                        self.test_set.batch_size))
        # Set and distribute events
        if events is None:
            events = DefaultTrainerEvents(epochs)
        if collect_all_times:
            events.append(SummaryGeneratorInferenceEvent(stats))
        else:
            events.append(SummaryGeneratorEvent(stats))
        executor_events = [e for e in events if isinstance(e, ExecutorEvent)]
        optimizer_events = [e for e in events if isinstance(e, OptimizerEvent)]
        sampler_events = [e for e in events if isinstance(e, SamplerEvent)]
        events = [e for e in events if isinstance(e, RunnerEvent)]

        # Append events to executor and samplers
        self.executor.events.extend(executor_events)
        self.train_set.events.extend(sampler_events)
        if self.test_set is not None:
            self.test_set.events.extend(sampler_events)

        try:
            for event in events: event.before_training(self, stats)

            # Run test set prior to training
            self._test_accuracy(stats, events)

            for epoch in range(epochs):
                for event in events: event.before_epoch(epoch, self, stats)
                self._train(stats, events, optimizer_events)
                self._test_accuracy(stats, events)
                for event in events: event.after_epoch(epoch, self, stats)

        except (StopIteration, StopTraining):
            pass # If stopping was requested

        for event in events: event.after_training(self, stats)
              
        # Remove events from executor and samplers
        del self.executor.events[-len(executor_events):]
        del self.train_set.events[-len(sampler_events):]
        if self.test_set is not None:
            del self.test_set.events[-len(sampler_events):]

        return stats
