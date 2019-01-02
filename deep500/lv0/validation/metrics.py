from deep500.utils.metric import TestMetric
from deep500.utils.metrics import L2Error, MaxError, DiffHeatmap, WallclockTime

_DEFAULT_OP_METRIC_CLASSES = [L2Error, MaxError, DiffHeatmap, WallclockTime]


def DefaultOpMetrics():
    return [metric() for metric in _DEFAULT_OP_METRIC_CLASSES]
