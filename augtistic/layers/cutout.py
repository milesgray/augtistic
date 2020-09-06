import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomCutout(Layer):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to images.
    Uses TensorFlow Addons tfa.image.random_cutout under hood.
    https://www.tensorflow.org/addons/api_docs/python/tfa/image/random_cutout
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        cutout_size: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The contrast factor will be randomly picked between
            [1.0 - lower, 1.0 + upper].
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, cutout_size, replace=0, seed=None, name=None, **kwargs):
        self.cutout_size = cutout_size        
        if isinstance(cutout_size, (tuple, list)):
            self.cutout_height = cutout_size[0]
            self.cutout_width = cutout_size[1]
        else:
            self.cutout_height = self.cutout_width = cutout_size
        if self.cutout_height < 0. or self.cutout_width < 0.:
            raise ValueError('Cutout size cannot have negative values,'
                             ' got {}'.format(factor))
        self.replace = replace
        if self.replace < 0.:
            raise ValueError('Replace cannot have negative value,'
                             ' got {}'.format(factor))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        
        super(RandomCutout, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_cutout_inputs():
            return tfa.image.random_cutout(inputs, 
                                           mask_size=self.cutout_size, 
                                           constant_values=self.replace, 
                                           seed=self.seed)

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
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))