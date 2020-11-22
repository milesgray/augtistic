import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.rand as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomHue(Layer):
    """Adjust the hue of an image or images by a random delta.
    It converts an RGB image to float representation, 
    converts it to HSV, adds an offset to the hue channel, 
    converts back to RGB and then back to the original data type.    
    
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Attributes:
        max_delta: float. The maximum value for the random delta. Must be in the interval [0, 0.5].
        seed: Integer. Used to create a random seed. Pass None to make determinalistic.
        name: A string, the name of the layer.
    
    Raise:
        ValueError: if max_delta is invalid.
    """

    def __init__(self, max_delta, seed=random.randint(0,1000), name=None, **kwargs):
        self.max_delta = max_delta

        if self.max_delta < 0. or self.max_delta > 0.5:
            raise ValueError('max_delta cannot have negative values or greater than 0.5,'
                             ' got {}'.format(max_delta))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomHue, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_hue_inputs():
            return tf.image.random_hue(inputs, self.max_delta, self.seed)

        output = tf_utils.smart_cond(training, random_hue_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'max_delta': self.max_delta,
            'seed': self.seed,
        }
        base_config = super(RandomHue, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))