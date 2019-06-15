""" A collection of general-purpose metrics for Deep500. """

import numpy as np
import time
from deep500.utils.metric import TestMetric

# General purpose accuracy metrics
##################################

class NormDifference(TestMetric):
    def __init__(self, ord):
        """ Returns the difference between all tensors as a list of norms.
        @param ord Order of the norm. Abides by numpy's definition of norm
                   order. See numpy.linalg.norm documentation.
        """
        self.ord = ord

    def measure(self, inputs, outputs, reference_outputs):
        if isinstance(outputs, (list, tuple)):
            assert len(outputs) == len(reference_outputs)
            return [np.linalg.norm((ro - o).flatten(), ord=self.ord) for o, ro
                    in zip(outputs, reference_outputs)]
        else:
            return np.linalg.norm((outputs - reference_outputs).flatten(),
                                  ord=self.ord)


class L1Error(NormDifference):
    def __init__(self):
        super().__init__(1)


class L2Error(NormDifference):
    def __init__(self):
        super().__init__(2)


class MaxError(NormDifference):
    def __init__(self):
        super().__init__(np.inf)


class DiffHeatmap(TestMetric):
    def __init__(self, filename='heatmap.png'):
        super().__init__()
        self._filename = filename

    def measure(self, inputs, outputs, reference_outputs):
        if isinstance(outputs, (list, tuple)):
            assert len(outputs) == len(reference_outputs)
            difference = [2 * (o - ro) / (abs(o) + abs(ro) + 1e-12)
                          for o, ro in zip(outputs, reference_outputs)]
        else:
            difference = 2 * (outputs - reference_outputs) / (
                abs(outputs) + abs(reference_outputs) + 1e-12)

        return difference

    def measure_summary(self, inputs, outputs, reference_outputs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # Necessary for projection='3d'

        result = self.measure(inputs, outputs, reference_outputs)

        if isinstance(result, (list, tuple)):
            nplots = len(result)
        else:
            nplots = 1

        # Draw heat-maps
        fig = plt.figure()

        if nplots == 1 and not isinstance(result, (list, tuple)):
            diff = [result]
        else:
            diff = result

        for i, data in enumerate(diff):

            if len(data.shape) > 2:
                data = np.squeeze(data)
            if len(data.shape) < 2:
                data = np.reshape(data, (data.shape[0], 1))

            if len(data.shape) > 3:
                raise ValueError('Too many dimensions for data')
            if len(data.shape) > 2:
                ax = fig.add_subplot(1, len(diff), i + 1, projection='3d')
                xs = []
                ys = []
                zs = []
                for z in range(data.shape[0]):
                    for y in range(data.shape[1]):
                        xs += list(range(data.shape[2]))
                        ys += [y] * data.shape[2]
                    zs += [z] * data.shape[1] * data.shape[2]
                im = ax.scatter(
                    xs, ys, zs,
                    c=np.reshape(data, (1, data.size))[0],
                    cmap="PuOr", vmin=-2., vmax=2.
                )
            else:
                ax = fig.add_subplot(1, len(diff), i + 1)
                im = ax.imshow(data, cmap="PuOr", vmin=-2., vmax=2.)

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(ylabel="Relative Percent Difference",
                               rotation=-90, va="bottom")

            # We want to show all ticks...
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))

        plt.savefig(self._filename)
        return self._filename


class VarHeatmap(TestMetric):
    def __init__(self, filename='heatmap', reruns=10):
        super().__init__()
        self._filename = filename
        self._reruns = reruns
        self._results = []

    @property
    def reruns(self):
        return self._reruns

    def end(self, outputs):
        self._results.append(outputs)

    def measure(self, inputs, outputs, reference_outputs):
        if not self._results:
            raise ValueError
        if isinstance(self._results[0], (list, tuple)):
            variance = []
            num_sub_res = len(self._results[0])
            cmp_str = '('
            for i in range(num_sub_res):
                comb_array = np.copy(self._results[0][i])
                comb_array = np.reshape(comb_array, (1, *comb_array.shape))
                cmp_res = True
                for j in range(1, len(self._results)):
                    second = np.reshape(self._results[j][i],
                                        (1, *self._results[j][i].shape))
                    np.concatenate((comb_array, second), axis=0)
                    if not np.array_equal(self._results[0][i],
                                          self._results[j][i]):
                        cmp_res = False
                variance.append(np.var(comb_array, axis=0))
                cmp_str += str(cmp_res) + ', '
            cmp_str += ')'
            print(cmp_str)
        else:
            comb_array = np.copy(self._results[0])
            comb_array = np.reshape(comb_array, (1, *comb_array.shape))
            cmp_res = True
            for i in range(1, len(self._results)):
                second = np.reshape(self._results[i],
                                    (1, *self._results[i].shape))
                np.concatenate((comb_array, second), axis=0)
                if not np.array_equal(self._results[0],
                                      self._results[i]):
                    cmp_res = False
            variance.append(np.var(comb_array, axis=0))
            print(cmp_res)

        return variance

    def measure_summary(self, inputs, outputs, reference_outputs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # Necessary for projection='3d'

        result = self.measure(inputs, outputs, reference_outputs)
        for i, res in enumerate(result):
            np.savetxt('{}_{:03d}.out'.format(self._filename, i),
                       np.squeeze(res), delimiter=',')

        if isinstance(result, (list, tuple)):
            nplots = len(result)
        else:
            nplots = 1

        # Draw heat-maps
        fig = plt.figure()

        if nplots == 1 and not isinstance(result, (list, tuple)):
            diff = [result]
        else:
            diff = result

        for i, data in enumerate(diff):

            if len(data.shape) > 2:
                data = np.squeeze(data)
            if len(data.shape) < 2:
                data = np.reshape(data, (data.shape[0], 1))

            if len(data.shape) > 3:
                raise ValueError
            if len(data.shape) > 2:
                ax = fig.add_subplot(1, len(diff), i + 1, projection='3d')
                xs = []
                ys = []
                zs = []
                for z in range(data.shape[0]):
                    for y in range(data.shape[1]):
                        xs += list(range(data.shape[2]))
                        ys += [y] * data.shape[2]
                    zs += [z] * data.shape[1] * data.shape[2]
                im = ax.scatter(
                    xs, ys, zs,
                    c=np.reshape(data, (1, data.size))[0],
                    cmap="YlOrBr", vmin=0.,
                    vmax=max(1., np.max(data)-np.min(data))
                )
            else:
                ax = fig.add_subplot(1, len(diff), i + 1)
                im = ax.imshow(data, cmap="YlOrBr", vmin=0.,
                               vmax=max(1., np.max(data)-np.min(data)))

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(ylabel="Variance",
                               rotation=-90, va="bottom")

            # We want to show all ticks...
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))

        plt.savefig(self._filename + '.png')
        return self._filename


# General purpose timing metrics
################################

class WallclockTime(TestMetric):
    """ Low-accuracy wall-clock timer """

    def __init__(self, reruns=100, avg_over=10):
        """ Runs the test "reruns" times, returning all the runtimes averaged
            over "avg_over" consecutive runs. Summary returns the median value
            of the average wallclock times.
            @param reruns Total number of times to run test
            @param avg_over Number of consecutive runs to average over
        """
        if reruns < 0:
            raise ValueError('Number of reruns must be non-negative')
        if avg_over < 0:
            raise ValueError('Number of runs to average over must be non-negative')
        if avg_over > reruns and reruns > 0:
            avg_over = reruns
        self._reruns = reruns
        self._avg_over = avg_over
        self._begintime = []
        self._endtime = []
        self._t = 0

    @property
    def reruns(self):
        return self._reruns

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return False

    def begin(self, *args):
        if self._t % self._avg_over == 0:
            self._begintime.append(time.time())

    def end(self, *args):
        self._t += 1
        if self._t % self._avg_over == 0:
            self._endtime.append(time.time())

    def measure(self, inputs, outputs, reference_outputs):
        return [(e - b) / self._avg_over for b, e in zip(self._begintime, self._endtime)]

    def measure_summary(self, inputs, outputs, reference_outputs):
        result = np.median(self.measure(inputs, outputs, reference_outputs))
        # Pretty-print time duration
        if result >= 10:  # Seconds
            return '%.2f seconds' % float(result)
        elif result >= 10 * 1e-3:  # Milliseconds
            return '%.2f ms' % float(result * 1e3)
        elif result >= 1 * 1e-6:  # Microseconds
            return '%.2f us' % float(result * 1e6)
        else:  # Nanoseconds
            return '%f ns' % float(result * 1e9)
