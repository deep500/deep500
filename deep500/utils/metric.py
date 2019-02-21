from typing import List, Any


class TestMetric(object):

    def begin(self, inputs: List[Any]) -> None:
        """ Optional method called before every time the test begins.
            Used e.g., for timing metrics.
            @param inputs A list of inputs of the tested method.
            @return None
        """
        pass

    def end(self, outputs: List[Any]) -> None:
        """ Optional method called right after every time the test ends.
            Used e.g., for timing metrics.
            @param outputs A list of outputs of the tested method.
            @return None
        """
        pass

    @property
    def reruns(self) -> int:
        """ Returns 1 or larger if this metric requires N full reruns for
            measuring this metric. Typically used in timing and repeatability
            analyses, in which any other metric would interfere with the
            profiling. If 0 is returned, the metric will use the original test
            run.
        """
        return 0

    @property
    def requires_inputs(self) -> bool:
        """ Set to True if this metric depends on the inputs of the measured
            method in the `begin()` call. """
        return True

    @property
    def requires_outputs(self) -> bool:
        """ Set to True if this metric depends on the outputs of the measured
            method in the `end()` call. """
        return True

    def measure(self, inputs: List[Any], outputs: List[Any],
                reference_outputs: List[Any]) -> Any:
        """ Returns the metric's results after the full test ends.
            @param inputs A list of inputs of the tested method.
            @param outputs A list of outputs of the tested method.
            @param reference_outputs A list of reference outputs given by the
                                     test.
            @return The metric (which could be of any type).
        """
        raise NotImplementedError('Abstract Class')

    def measure_summary(self, inputs: List[Any], outputs: List[Any],
                        reference_outputs: List[Any]) -> str:
        """ Returns the metric's results as a human-readable summary string,
            after the full test ends.
            @param inputs A list of inputs of the tested method.
            @param outputs A list of outputs of the tested method.
            @param reference_outputs A list of reference outputs given by the
                                     test.
            @return The metric summary as a string.
        """
        return str(self.measure(inputs, outputs, reference_outputs))
