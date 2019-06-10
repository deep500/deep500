from typing import Any, List, Optional, Union

from deep500 import (Dataset, DefaultTrainerEvents,
                     GraphExecutor, Sampler,
                     ShuffleSampler, Optimizer, Trainer, TestMetric, TrainingEvent)
from deep500.lv2.validation.metrics import DefaultTrainingMetrics

SamplerOrDataset = Union[Sampler, Dataset]

def test_training(executor: GraphExecutor, training_set: SamplerOrDataset,
                  validation_set: Optional[SamplerOrDataset], optimizer: Optimizer, 
                  epochs: int, batch_size: int, output_node: Optional[str] = None,
                  metrics: List[TestMetric] = DefaultTrainingMetrics(),
                  events: List[TrainingEvent] = None) -> List[Any]:
    """ Tests training for a given number of epochs and reports metrics.
        @param executor The graph executor to run.
        @param training_set The training set sampler or dataset to use. 
                            If Dataset given, uses a ShuffleSampler.
        @param validation_set The test set sampler or dataset to use. Can be None.
                              If Dataset given, uses a ShuffleSampler.
        @param optimizer The optimization function to use.
        @param epochs The number of epochs to run for.
        @param batch_size The batch size to use.
        @param output_node The name of the network's output node.
        @param metrics Metrics to measure. Some metrics can also inherit from
                       TrainingEvent to obtain more information (see TestAccuracy
                       for example).
        @param events A list of events to use for the executor, optimizer, and 
                      Trainer. If some metrics are also events, there is no need
                      to include them here as well, they will be added automatically.
        @return A list of objects resulting from the metrics.
    """
    # Default events
    if events is None:
        events = DefaultTrainerEvents(epochs)

    # Create Samplers according to inputs
    if isinstance(training_set, Sampler):
        tsampler = training_set
    elif isinstance(training_set, Dataset):
        tsampler = ShuffleSampler(training_set, batch_size)
    else:
        raise ValueError('Training set type not supported')
    if validation_set is None:
        vsampler = None
    elif isinstance(validation_set, Sampler):
        vsampler = validation_set
    elif isinstance(validation_set, Dataset):
        vsampler = ShuffleSampler(validation_set, batch_size)
    else:
        raise ValueError('Training set type not supported')

    # Create trainer
    trainer = Trainer(tsampler, vsampler, executor, optimizer, output_node)

    # Add metrics to events as necessary
    for metric in metrics:
        if isinstance(metric, TrainingEvent):
            events.append(metric)

    # Run training with metrics
    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for metric in normal_metrics:
        metric.begin(input)
    outputs = trainer.run_loop(epochs, events)
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            metric.begin(input)
            outputs = trainer.run_loop(epochs, events)
            metric.end(outputs)

    # Execute metrics
    results = []
    for metric in metrics:
        result = metric.measure(trainer, outputs, None)
        results.append(result)
        summary = metric.measure_summary(trainer, outputs, None)
        print("{}: {}".format(
                type(metric).__name__, summary))

    return results
