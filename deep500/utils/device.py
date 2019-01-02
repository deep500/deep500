import abc


class DeviceType(metaclass=abc.ABCMeta):
    def __init__(self, num: int):
        # specify the device number gpu:0, gpu:1, cpu:0, etc
        self.num = num

    @abc.abstractmethod
    def is_gpu(self):
        pass


class GPUDevice(DeviceType):
    def __init__(self, num: int = 0):
        super(GPUDevice, self).__init__(num)

    def is_gpu(self):
        return True


class CPUDevice(DeviceType):
    def __init__(self, num: int = 0):
        super(CPUDevice, self).__init__(num)

    def is_gpu(self):
        return False
