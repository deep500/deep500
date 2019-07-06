import os
import random
import numpy as np
from typing import List

from deep500.lv2.dataset import Dataset
from deep500.lv2.sampler import Sampler
from deep500.lv2.event import SamplerEvent
from deep500.utils.onnx_interop.losses import SoftmaxCrossEntropy

# NOTE: This file is heavily based on the TensorFlow official ImageNet 
# preprocessing pipeline, which can be found at:
# https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py

# ImageNet dataset statistics
_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
_NUM_TRAIN_IMAGES = 1281167
_NUM_VALIDATION_IMAGES = 50000
_SHUFFLE_BUFFER = 10000
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256

def imagenet_shape(is_nchw=True):
    if is_nchw:
        return (_NUM_CLASSES, _NUM_CHANNELS, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)
    else:
        return (_NUM_CLASSES, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)

def imagenet_loss():
    return SoftmaxCrossEntropy


class TFRecordD500Dataset(Dataset):
    def __init__(self, files: List[str], sample_node: str, label_node: str,
                 total_samples: int, batch_size: int, 
                 shuffle: bool = True, shuffle_buffer: int = _SHUFFLE_BUFFER,
                 augment: bool = True, is_nchw: bool = True, seed: int = None):
        try:
            import tensorflow as tf
        except (ImportError, ModuleNotFoundError) as ex:
            raise ImportError('Cannot use TFRecordDataset without TensorFlow: %s' % str(ex))

        # Create a TensorFlow session without consuming GPU memory
        config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        random.shuffle(files)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.prefetch(buffer_size=shuffle_buffer)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda value: _parse_example_proto(value, augment, is_nchw),
                batch_size=batch_size,
                num_parallel_batches=1,
            drop_remainder=True))

        dataset = dataset.repeat()
        self.records = dataset
        self.record_iter = dataset.make_initializable_iterator()
        self.riter_op = self.record_iter.get_next()
        self.samples = total_samples
        self.input_node = sample_node
        self.label_node = label_node

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        images, labels = self.sess.run(self.riter_op)
        return {self.input_node: images, self.label_node: labels}

    def reset(self):
        self.sess.run(self.record_iter.initializer)


class SyntheticDataset(Dataset):
    """ Creates synthetic data to replace imagenet. """
    def __init__(self, input_node, label_node, num_images, batch_size,
                 seed=None):
        self.length = num_images
        self.input_node = input_node
        self.label_node = label_node
        self.batch_size = batch_size
        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        sample = np.random.rand(self.batch_size,
                                _NUM_CHANNELS,
                                _DEFAULT_IMAGE_SIZE,
                                _DEFAULT_IMAGE_SIZE).astype(np.float32)
        label = np.random.randint(0, _NUM_CLASSES, (self.batch_size,),
                                  dtype=np.int64)
        return {self.input_node: sample, self.label_node: label}

    def __len__(self):
        return self.length

    def reset(self):
        pass

def load_imagenet(input_node: str, label_node: str, batch_size: int, *args, **kwargs):
    train_sampler = TFRecordImageNetSampler(input_node, label_node, batch_size, *args, **kwargs)
    validation_sampler = TFRecordImageNetSampler(input_node, label_node, batch_size, *args, 
                                                 is_training=False, shuffle=False, augment=False, **kwargs)
    return train_sampler, validation_sampler

class TFRecordImageNetSampler(Sampler):
    def __init__(
        self,
        sample_node: str, label_node: str,
        batch_size: int,
        imagenet_path: str = None,
        is_training: bool = True,
        shuffle: bool = True,
        shuffle_buffer: int = _SHUFFLE_BUFFER,
        augment: bool = True,
        is_nchw: bool = True,
        synthetic: bool = False,
        seed: int = None,
        events: List[SamplerEvent] = None
    ):
        # Create the dataset
        if synthetic is False and imagenet_path is None:
            if 'IMAGENET_PATH' not in os.environ:
                raise ValueError('ImageNet path not given and the IMAGENET_PATH'
                                 ' environment variable is not set. Please set'
                                 ' or use synthetic=True')
            imagenet_path = os.environ['IMAGENET_PATH']

        if synthetic is False:
            all_files = os.listdir(imagenet_path)
            if is_training:
                files = [os.path.join(imagenet_path, f) for f in all_files
                         if f.startswith('train-')]
            else:
                files = [os.path.join(imagenet_path, f) for f in all_files
                         if f.startswith('validation-')]

        num_images = (_NUM_TRAIN_IMAGES if is_training else
                      _NUM_VALIDATION_IMAGES)

        if synthetic is True:
            self.dataset = SyntheticDataset(sample_node, label_node, num_images,
                                            batch_size, seed)
        else:
            self.dataset = TFRecordD500Dataset(files, sample_node, label_node,
                                               num_images, batch_size, shuffle,
                                               shuffle_buffer, augment, is_nchw,
                                               seed)
        self.batch_size = batch_size
        self.seed = seed
        self.events = events or []
        self.as_op = False
        self.cnt = 0
        self.reset()

    def as_operator(self):
        """ Returns a TensorFlow Operation that generates the input and label,
            to streamline data serving.
        """
        # TODO(talbn): Add event processing as tf.py_func nodes (especially the "before sampling" event)
        self.as_op = True
        return self.dataset.records.make_one_shot_iterator().get_next()

    def __iter__(self):
        return self

    def __next__(self):
        if self.as_op:
            return None
        if self.cnt >= len(self.dataset) // self.batch_size:
            raise StopIteration
        self.cnt += 1


        for event in self.events: event.before_sampling(self, self.batch_size)
        sample = self.dataset[0] # Returns a minibatch in any case
        for event in self.events: event.after_sampling(self, self.batch_size)
        return sample
        
    def __call__(self):
        return self.__next__()

    def __len__(self):
        """ Defines the length of an epoch, or 0 for running until a 
            StopIteration exeption is raised. """
        return len(self.dataset) // self.batch_size

    def reset(self):
        self.dataset.reset()
        self.cnt = 0


########################################################################
########################################################################
########################################################################
# Parses the TF records
# Adapted from https://www.github.com/tensorflow/models

def _parse_example_proto(example_serialized, augment, is_nchw):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  import tensorflow as tf
  
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int64) - 1

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  image_buffer = features['image/encoded']

  #### PREPROCESS

  if augment:
    # For training, we want to randomize some of the distortions.
    image = _decode_crop_and_flip(image_buffer, bbox, _NUM_CHANNELS)
    image = _resize_image(image, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)
  else:
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=_NUM_CHANNELS)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)

  image.set_shape([_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])

  image = _mean_image_subtraction(image, _CHANNEL_MEANS, _NUM_CHANNELS)
  if is_nchw:
      image = tf.transpose(image, [2, 0, 1])

  return image, label



def _decode_crop_and_flip(image_buffer, bbox, num_channels):
  """Crops the given image to a random part of the image, and randomly flips.

  We use the fused decode_and_crop op, which performs better than the two ops
  used separately in series, but note that this requires that the image be
  passed in as an un-decoded string Tensor.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.

  Returns:
    3-D tensor with cropped image.

  """
  import tensorflow as tf
  # A large fraction of image datasets contain a human-annotated bounding box
  # delineating the region of the image containing the object of interest.  We
  # choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.image.extract_jpeg_shape(image_buffer),
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
  cropped = tf.image.decode_and_crop_jpeg(
      image_buffer, crop_window, channels=num_channels)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped

def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  import tensorflow as tf
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  import tensorflow as tf
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)

  return image - means


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  import tensorflow as tf
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  import tensorflow as tf
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  import tensorflow as tf
  return tf.image.resize_images(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)
