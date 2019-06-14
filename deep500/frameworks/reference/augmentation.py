""" File containing various reference data augmentation transformations for
    Deep500 Samplers. """

from typing import Any, Dict, Tuple
import numpy as np
import random
import PIL.Image


class DataAugmentation(object):
    """ Represents a general data augmentation for supervised learning. """

    def __init__(self, input_node: str, label_node: str):
        self.input = input_node
        self.label = label_node


class SingleSampleAugmentation(DataAugmentation):
    """ Data augmentation class that works individually per sample. """
    def __init__(self, input_node: str, label_node: str):
        super().__init__(input_node, label_node)

    def augment_sample(self, sample: Any, label: Any):
        """ Augments a single sample.
            @param sample: The sample to augment.
            @param label: The corresponding label.
            @return: A 2-tuple of augmented (sample, label).
        """
        raise NotImplementedError

    def __call__(self, batch: Dict[str, np.ndarray]):
        samples = batch[self.input]
        labels = batch[self.label]
        for i in range(samples.shape[0]):
            samples[i], labels[i] = self.augment_sample(samples[i], labels[i])
        return batch


class ReplicateBatch(DataAugmentation):
    """ Replicates samples and labels in minibatch for a specified number of
        times. """
    def __init__(self, input_node: str, label_node: str, duplicates: int):
        super().__init__(input_node, label_node)
        self.duplicates = duplicates

    def __call__(self, batch: Dict[str, np.ndarray]):
        replicate_in = [1] * len(batch[self.input].shape)
        replicate_in[0] = self.duplicates
        replicate_lbl = [1] * len(batch[self.label].shape)
        replicate_lbl[0] = self.duplicates

        return {self.input: np.tile(batch[self.input], replicate_in),
                self.label: np.tile(batch[self.label], replicate_lbl)}


class Crop(SingleSampleAugmentation):
    """ Random/center image crop augmentation with optional constant
        padding. """
    def __init__(self, input_node: str, label_node: str,
                 crop_size: Tuple[int, int],
                 resize: bool = False, random_crop: bool = False,
                 padding: Tuple[int, int] = None, fill: Any = 0):
        super().__init__(input_node, label_node)
        self.crop_size = crop_size
        self.resize = resize
        self.random_crop = random_crop
        self.padding = padding
        self.fill = fill

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        c, h, w = sample.shape

        # Pad, if specified
        if self.padding is not None:
            pad_tuple = [(0, 0)] + [(p, p) for p in self.padding]
            sample = np.pad(sample, pad_tuple, 'constant',
                            constant_values=self.fill)

        crop_y, crop_x = self.crop_size

        # Crop
        if self.random_crop:  # Crop randomly
            y = random.randint(0, h - crop_y)
            x = random.randint(0, w - crop_x)
        else:  # Center crop
            y = h // 2 - crop_y // 2
            x = w // 2 - crop_x // 2
        sample = sample[:, y:y + crop_y, x:x + crop_x]

        # Resize back, if specified
        if self.resize:
            # Convert back to image (expensive)
            image = PIL.Image.fromarray(
                (sample.transpose(1, 2, 0) * 256).astype(np.uint8), 'RGB')
            resized = image.resize((w, h))
            sample = np.array(resized).astype(sample.dtype) / 256.0
            sample = sample.transpose(1, 2, 0)

        return sample, label


class RandomFlip(SingleSampleAugmentation):
    """ Flip an input randomly according to a specific axis and probability. """

    def __init__(self, input_node: str, label_node: str, axis: int = -1,
                 p: float = 0.5):
        super().__init__(input_node, label_node)
        self.p = p
        self.axis = axis

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        if random.random() > self.p:
            return np.flip(sample, self.axis), label
        return sample, label


class Cutout(SingleSampleAugmentation):
    """ Randomly masks out one or more patches from an image.
        Adapted from https://github.com/uoguelph-mlrg/Cutout
        Args:
            holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, input_node: str, label_node: str, holes: int,
                 length: int):
        super().__init__(input_node, label_node)
        self.holes = holes
        self.length = length

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        c, h, w = sample.shape
        mask = np.ones((h, w), sample.dtype)

        for n in range(self.holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        return (sample[:, :, :] * mask), label
