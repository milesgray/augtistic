import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomSaturation(Layer):
    """Adjust the saturation of an image or images by a random factor.
    Saturation is adjusted independently for each channel of each image during
    training.
    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * saturation_factor + mean`.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The saturation factor will be randomly picked between
            [1.0 - lower, 1.0 + upper].
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, factor, seed=None, name=None, **kwargs):
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor
        if self.lower < 0. or self.upper < 0. or self.lower > 1.:
            raise ValueError('Factor cannot have negative values or greater than 1.0,'
                             ' got {}'.format(factor))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomSaturation, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_saturation_inputs():
            return tf.image.random_saturation(image=inputs, 
                                              lower=1. - self.lower, 
                                              upper=1. + self.upper,
                                              seed=self.seed)

        output = tf_utils.smart_cond(training, random_saturation_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'factor': self.factor,
            'seed': self.seed,
        }
        base_config = super(RandomSaturation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
