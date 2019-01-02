from deep500 import Event

class ExecutorEvent(Event):
    def before_executor(self, inputs):
        pass

    def after_inference(self, inference_results):
        pass

    def after_backprop(self, gradients):
        pass
