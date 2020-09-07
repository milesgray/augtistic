import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomMeanFilter2D(Layer):
    """Perform median filtering on image(s).
    Uses ref:tfa.image.median_filter2d as filter algorithm.
    https://www.tensorflow.org/addons/api_docs/python/tfa/image/median_filter2d
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        filter_factor: A positive int, the amount to divide image height by to determine `filter_size` for gaussian filter
        padding: A string, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of padding algorithm to use, which is compatible with mode argument in tf.pad. For more details, please refer to https://www.tensorflow.org/api_docs/python/tf/pad
        constant_values: A scalar, the pad value to use in "CONSTANT" padding mode.
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, filter_factor=10, padding='REFLECT', constant_values=0, seed=random.randint(0,1000), name=None, **kwargs):
        self.filter_factor = filter_factor        
        if self.filter_factor < 0:
            raise ValueError('Filter factor cannot have negative values,'
                             ' got {}'.format(factor))
        self.constant_values = constant_values
        self.padding = padding
        if self.padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError('Padding must be one of "REFELCT", "CONSTANT", or "SYMMETRIC",'
                             ' got {}'.format(padding))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        self._rng = make_generator(self.seed)
        super(RandomMeanFilter2D, self).__init__(name=name, **kwargs)
    
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
    
    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_gaussian_inputs():
            blend = self._rng.uniform([], 0.1, 1.0, dtype=tf.float32)
            filtered_inputs = tfa.image.mean_filter2d(inputs, 
                                               filter_shape=self.height//self.filter_factor,
                                               padding=self.padding,
                                               constant_values=self.constant_values)
            return tfa.image.blend(filtered_inputs, inputs, blend)

        output = tf_utils.smart_cond(training, random_gaussian_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cutout_size': self.cutout_size,
            'seed': self.seed,
        }
        base_config = super(RandomMeanFilter2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomMedianFilter2D(Layer):
    """Perform median filtering on image(s).
    Uses ref:tfa.image.median_filter2d as filter algorithm.
    https://www.tensorflow.org/addons/api_docs/python/tfa/image/median_filter2d
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        filter_factor: A positive int, the amount to divide image height by to determine `filter_size` for gaussian filter
        padding: A string, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of padding algorithm to use, which is compatible with mode argument in tf.pad. For more details, please refer to https://www.tensorflow.org/api_docs/python/tf/pad
        constant_values: A scalar, the pad value to use in "CONSTANT" padding mode.
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, filter_factor=10, padding='REFLECT', constant_values=0, seed=random.randint(0,1000), name=None, **kwargs):
        self.filter_factor = filter_factor        
        if self.filter_factor < 0:
            raise ValueError('Filter factor cannot have negative values,'
                             ' got {}'.format(factor))
        self.constant_values = constant_values
        self.padding = padding
        if self.padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError('Padding must be one of "REFELCT", "CONSTANT", or "SYMMETRIC",'
                             ' got {}'.format(padding))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        self._rng = make_generator(self.seed)
        super(RandomMedianFilter2D, self).__init__(name=name, **kwargs)
    
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
    
    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_gaussian_inputs():
            blend = self._rng.uniform([], 0.1, 1.0, dtype=tf.float32)
            filtered_inputs = tfa.image.median_filter2d(inputs, 
                                               filter_shape=self.height//self.filter_factor,
                                               padding=self.padding,
                                               constant_values=self.constant_values)
            return tfa.image.blend(filtered_inputs, inputs, blend)

        output = tf_utils.smart_cond(training, random_gaussian_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cutout_size': self.cutout_size,
            'seed': self.seed,
        }
        base_config = super(RandomMedianFilter2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomGaussian2D(Layer):
    """Perform Gaussian blur on image(s).
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        filter_factor: A positive int, the amount to divide image height by to determine `filter_size` for gaussian filter
        padding: A string, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of padding algorithm to use, which is compatible with mode argument in tf.pad. For more details, please refer to https://www.tensorflow.org/api_docs/python/tf/pad
        replace: A scalar, the pad value to use in "CONSTANT" padding mode.
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, probability, padding='REFLECT', replace=0, seed=None, name=None, **kwargs):
        self.probability = probability        
        if self.probability < 0:
            raise ValueError('probability cannot have negative values,'
                             ' got {}'.format(probability))
        self.replace = replace
        if self.replace < 0.:
            raise ValueError('Replace cannot have negative value,'
                             ' got {}'.format(replace))
        self.padding = padding
        if self.padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError('Replace must be one of "REFELCT", "CONSTANT", or "SYMMETRIC",'
                             ' got {}'.format(padding))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        self._rng = make_generator(self.seed)
        super(RandomGaussian2D, self).__init__(name=name, **kwargs)
    
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
    
    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        output = self._random_blur(inputs, 
                                  height=self.height, 
                                  width=self.width, 
                                  p=self.probability)
        output.set_shape(inputs.shape)
        return output

    def _gaussian_blur(self, image, kernel_size, sigma, padding='SAME'):
        """Blurs the given image with separable convolution.
        Args:
            image: Tensor of shape [height, width, channels] and dtype float to blur.
            kernel_size: Integer Tensor for the size of the blur kernel. This is should
            be an odd number. If it is an even number, the actual kernel size will be
            size + 1.
            sigma: Sigma value for gaussian operator.
            padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
        Returns:
            A Tensor representing the blurred image.
        """
        radius = tf.cast(kernel_size / 2, dtype=tf.dtypes.int32)
        kernel_size = radius * 2 + 1
        x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.dtypes.float32)
        blur_filter = tf.exp(
            -tf.pow(x, 2.0) / 
            (2.0 * tf.pow(tf.cast(sigma, dtype=tf.dtypes.float32), 2.0)))
        blur_filter /= tf.reduce_sum(blur_filter)
        # One vertical and one horizontal filter.
        blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
        blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
        num_channels = tf.shape(image)[-1]
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        expand_batch_dim = image.shape.ndims == 3
        if expand_batch_dim:
            # Tensorflow requires batched input to convolutions, which we can fake with
            # an extra dimension.
            image = tf.expand_dims(image, axis=0)
        blurred = tf.nn.depthwise_conv2d(image, 
                                         blur_h, 
                                         strides=[1, 1, 1, 1], 
                                         padding=padding)
        blurred = tf.nn.depthwise_conv2d(blurred, 
                                         blur_v, 
                                         strides=[1, 1, 1, 1], 
                                         padding=padding)
        if expand_batch_dim:
            blurred = tf.squeeze(blurred, axis=0)
        return blurred

    def _random_blur(self, image, height, width, p=1.0):
        """Randomly blur an image.
        Args:
            image: `Tensor` representing an image of arbitrary size.
            height: Height of output image.
            width: Width of output image.
            p: probability of applying this transformation.
        Returns:
            A preprocessed image `Tensor`.
        """
        del width
        def _transform(image):
            sigma = self._rng.uniform([], 0.1, 1.5, dtype=tf.float32)
            return self._gaussian_blur(image, 
                                      kernel_size=height//10, 
                                      sigma=sigma, 
                                      padding='SAME')
        return self._random_apply(_transform, p=p, x=image)
        
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(self._rng.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x) 
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cutout_size': self.cutout_size,
            'seed': self.seed,
        }
        base_config = super(RandomGaussian2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))