from typing import Any, List, Optional
from deep500.utils.metric import TestMetric
from deep500.lv2.sampler import Sampler
from deep500.lv2.event import SamplerEvent

def test_sampler(sampler: Sampler, expected_output: Optional[Any] = None, 
                 metrics: List[TestMetric] = [], 
                 events: List[SamplerEvent] = []) -> List[Any]:
    """ Tests a Deep500 dataset sampler object.
        @param sampler The Sampler to test.
        @param expected_output An optional expected output.
        @param metrics Metrics to measure. Some metrics can also inherit from
                       SamplerEvent to obtain more information.
        @param events A list of events to use for the sampler. 
                      If some metrics are also events, there is no need
                      to include them here as well, they will be added automatically.
        @return A list of objects resulting from the metrics.
    """
    # Add metrics to events as necessary
    for metric in metrics:
        if isinstance(metric, SamplerEvent):
            events.append(metric)

    # Add metrics to sampler
    sampler.events.extend(events)

    # Run training with metrics
    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    try:
        for metric in normal_metrics:
            metric.begin(sampler)
        outputs = sampler()
        for metric in normal_metrics:
            metric.end(outputs)
    except StopIteration:
        outputs = None
        sampler.reset()

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            try:
                metric.begin(sampler)
                outputs = sampler()
                metric.end(outputs)
            except StopIteration:
                outputs = None
                sampler.reset()

    # Execute metrics
    results = []
    for metric in metrics:
        result = metric.measure(sampler, outputs, expected_output)
        results.append(result)
        summary = metric.measure_summary(sampler, outputs, expected_output)
        print("{}: {}".format(
                type(metric).__name__, summary))

    # Remove metrics from sampler
    del sampler.events[-len(events):]


    return results
