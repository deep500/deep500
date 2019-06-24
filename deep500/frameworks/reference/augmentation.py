""" File containing various reference data augmentation transformations for
    Deep500 Samplers. """

from typing import Any, Dict, Tuple, Union
import numpy as np
import random
import PIL.Image


class DataAugmentation(object):
    """ Represents a general data augmentation for supervised learning. """

    def __init__(self):
        self.input = None
        self.label = None

    def set_dataset_nodes(self, input_node: str, label_node: str = None):
        """ Resets dataset graph node names.
            @param input_node: The node where input data should be fed to.
            @param label_node: (optional) The target of the optimization
                               (e.g., label).
        """
        self.input = input_node
        self.label = label_node


class SingleSampleAugmentation(DataAugmentation):
    """ Data augmentation class that works individually per sample. """

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
        augmented = list(zip(*[self.augment_sample(s, l) for s, l in zip(samples, labels)]))

        return {self.input: np.asarray(augmented[0]),
                self.label: np.asarray(augmented[1])}


class ReplicateBatch(DataAugmentation):
    """ Replicates samples and labels in minibatch for a specified number of
        times. """
    def __init__(self, duplicates: int):
        super().__init__()
        self.duplicates = duplicates

    def __call__(self, batch: Dict[str, np.ndarray]):
        replicate_in = [1] * len(batch[self.input].shape)
        replicate_in[0] = self.duplicates
        replicate_lbl = [1] * len(batch[self.label].shape)
        replicate_lbl[0] = self.duplicates

        return {self.input: np.tile(batch[self.input], replicate_in),
                self.label: np.tile(batch[self.label], replicate_lbl)}


class Crop(SingleSampleAugmentation):
    """ Random/center crop augmentation with optional constant padding. """
    def __init__(self,
                 crop_size: Tuple[int, int],
                 resize: bool = False, random_crop: bool = False,
                 padding: Tuple[int, int] = None, fill: Any = 0):
        super().__init__()
        self.crop_size = crop_size
        self.resize = resize
        self.random_crop = random_crop
        self.padding = padding
        self.fill = fill
        if self.resize and len(sample.shape) == 2:
            raise ValueError('Cannot resize non-image sample')

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        dims = len(sample.shape)
        non_crop_dims = dims - len(self.crop_size)
        if non_crop_dims < 0:
            raise ValueError('Cropping too many dimensions')
        # Define slice without the dimensions to crop
        shape = list(sample.shape[non_crop_dims:])
        slice_ = [slice(None)] * non_crop_dims

        # Pad, if specified
        if self.padding is not None:
            pad_tuple = [(0, 0)] * non_crop_dims + [(p, p) for p in self.padding]
            sample = np.pad(sample, pad_tuple, 'constant',
                            constant_values=self.fill)
            for i, pad in enumerate(self.padding):
                shape[i] += 2 * pad

        # Crop
        if self.random_crop:  # Crop randomly
            startp = [random.randint(0, s - c) for s,c in zip(shape, self.crop_size)]
        else:  # Center crop
            startp = [s // 2 - c // 2 for s,c in zip(shape, self.crop_size)]
        slice_ += [slice(s, s + c) for s, c in zip(startp, self.crop_size)]
        sample = sample[tuple(slice_)]

        # Resize back, if specified
        if self.resize:
            # Convert back to image (expensive)
            image = PIL.Image.fromarray(
                (sample.transpose(1, 2, 0) * 256).astype(np.uint8), 'RGB')
            resized = image.resize((w, h))
            sample = np.array(resized).astype(sample.dtype) / 256.0
            sample = sample.transpose(1, 2, 0)

        return sample, label


class Resize(SingleSampleAugmentation):
    """ Resize an image to a specified size. """
    def __init__(self,
                 target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = target_size

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        c, h, w = sample.shape

        # Convert back to image (expensive)
        image = PIL.Image.fromarray(
            (sample.transpose(1, 2, 0) * 256).astype(np.uint8), 'RGB')
        resized = image.resize(self.target_size)
        sample = np.array(resized).astype(sample.dtype) / 256.0
        sample = sample.transpose(1, 2, 0)

        return sample, label


class Normalize(SingleSampleAugmentation):
    """ Normalize samples according to given channel-wise mean/stddev. """
    def __init__(self,
                 mean: Union[float, Tuple[float, ...]] = 0.0,
                 stddev: Union[float, Tuple[float, ...]] = 1.0):
        super().__init__()
        self.mean = mean
        self.stddev = stddev

    def augment_sample(self, sample: np.ndarray, label: np.ndarray):
        for dim in range(sample.shape[0]):
            sample[dim] -= self.mean[dim]
            sample[dim] /= self.stddev[dim]
        return sample, label


class RandomFlip(SingleSampleAugmentation):
    """ Flip an input randomly according to a specific axis and probability. """

    def __init__(self, axis: int = -1,
                 p: float = 0.5):
        super().__init__()
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
            random_fill (bool): If True, fills hole with random values.
    """

    def __init__(self, holes: int = 1, length: int = 16,
                 random_fill: bool = False):
        super().__init__()
        self.holes = holes
        self.length = length
        self.random = random_fill

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

        if self.random:
            masked = sample[:, :, :] * mask
            masked += (1. - mask) * (
                    np.random.rand(c, h, w) - 0.5)
            return masked, label
        else:
            return (sample[:, :, :] * mask), label
