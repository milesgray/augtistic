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

    def __init__(self, filter_factor=10, padding='REFLECT', constant_values=0, seed=None, name=None, **kwargs):
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
        base_config = super(RandomMeanFilter2D, self).get_config()
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

    def __init__(self, filter_factor=10, padding='REFLECT', replace=0, seed=None, name=None, **kwargs):
        self.filter_factor = filter_factor        
        if self.filter_factor < 0:
            raise ValueError('Filter factor cannot have negative values,'
                             ' got {}'.format(factor))
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

        def random_cutout_inputs():
            sigma = self._rng.uniform([], 0.1, 1.5, dtype=tf.float32)
            return tfa.image.gaussian_filter2d(inputs, 
                                               filter_shape=self.filter_shape, 
                                               sigma=sigma)

        output = tf_utils.smart_cond(training, random_cutout_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cutout_size': self.cutout_size,
            'seed': self.seed,
        }
        base_config = super(RandomGaussian2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
