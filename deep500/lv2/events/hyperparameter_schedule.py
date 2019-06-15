import inspect
from typing import Any, Callable, List, Tuple, Union

from ..event import RunnerEvent, OptimizerEvent
from ..summaries import TrainingStatistics


class HyperparameterSchedule(RunnerEvent, OptimizerEvent):
    """ An Event that enables modifying hyperparameters (e.g., learning rate)
        during training, either per epoch or per step. """

    def __init__(self, per_epoch,
                 **hyperparameters: Union[Callable[[int], Any],
                                          List[Tuple[int, Any]]]):
        """ Initializes a hyperparameter schedule Event.
            @param per_epoch: If True, evaluated every epoch, otherwise
                              evaluated every step.
            @param hyperparameters: A dictionary of hyperparameter name to
                                    either a list of (epoch/step, value), or a
                                    lambda function that accepts epoch/step
                                    and returns a value.
        """
        self._per_epoch = per_epoch
        self._optimizer = None
        self._step = 0
        self._list_hp = {k: sorted(v, key=lambda a: a[0])
                         for k, v in hyperparameters.items()
                         if isinstance(v, (list, tuple))}
        self._lambda_hp = {k: v for k, v in hyperparameters.items()
                           if inspect.isfunction(v)}

    def _check_hp(self, value):
        # List schedules
        to_change = {}
        for name, lhp in self._list_hp.items():
            index = -1
            while (index + 1) < len(lhp) and value >= lhp[index + 1][0]:
                index += 1
                continue
            # If we should change a parameter, change it
            if index >= 0:
                to_change[name] = lhp[index + 1:]
                self._optimizer.set_parameter(name, lhp[index][1])
        for k, v in to_change.items():
            self._list_hp[k] = v

        # Lambda schedules
        for name, lhp in self._lambda_hp.items():
            self._optimizer.set_parameter(name, lhp(value))

    def before_optimizer_step(self, executor, optimizer, inputs):
        if self._optimizer is None:
            self._optimizer = optimizer
        if self._per_epoch is False:
            self._check_hp(self._step)
            self._step += 1

    def before_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        if self._optimizer is None:
            return
        if self._per_epoch:
            self._check_hp(epoch)


class EpochHPSchedule(HyperparameterSchedule):
    """ An Event that enables modifying hyperparameters (e.g., learning rate)
        during training at every epoch. """
    def __init__(self, **hyperparameters: Union[Callable[[int], Any],
                                                List[Tuple[int, Any]]]):
        """ Initializes a epoch-wise hyperparameter schedule Event.
            @param hyperparameters: A dictionary of hyperparameter name to
                                    either a list of (epoch, value), or a
                                    lambda function that accepts epoch
                                    and returns a value.
        """
        super().__init__(True, **hyperparameters)


class StepHPSchedule(HyperparameterSchedule):
    """ An Event that enables modifying hyperparameters (e.g., learning rate)
        during training at every step. """
    def __init__(self, **hyperparameters: Union[Callable[[int], Any],
                                                List[Tuple[int, Any]]]):
        """ Initializes a step-wise hyperparameter schedule Event.
            @param hyperparameters: A dictionary of hyperparameter name to
                                    either a list of (step, value), or a
                                    lambda function that accepts the current
                                    step and returns a value.
        """
        super().__init__(False, **hyperparameters)
