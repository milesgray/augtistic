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
class ClipImageRange(Layer):
    """Apply tf.clip_by_value to the input, to restrict values to the appropriate
    range depending on how the image data is scaled. Will automatically assign one of
    (0., 1.), (-1., 1.), or (0, 255) if a tuple range is not provided and just a single
    number is instead.

    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Attributes:
        clip_values: A tuple that describes the range of acceptable values for
            image data.  Generally one of:
                (0., 1.) or (-1., 1.) or (0, 255) or (-127, 127)
            If only a single value is provided, one of those 4 values will be
            chosen based on some internal rules.
        name: A string, the name of the layer.
    
    Raise:
        ValueError: if clip_values is more than 2 values
    """

    def __init__(self, clip_values, name=None, **kwargs):
        self.clip_values = clip_values
        if not isinstance(clip_values, tuple) and not isinstance(clip_values, list):
            if isinstance(clip_values, float):
                if clip_values > 0:
                    self.clip_values = (-1., 1.)
                else:
                    self.clip_values = (0.,1.)  
            elif isinstance(clip_values, int):
                if clip_values > 127:
                    self.clip_values = (0, 255)
                else:
                    self.clip_values = (-127,127)
        if len(self.clip_values) > 2:
            raise ValueError("[ERROR]\t Clip Values range is more than 2 values, should be the lower and upper bound on pixel values!")
        
        self.input_spec = InputSpec(ndim=4)
        
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()
        
        output = tf.clip_by_value(inputs, self.clip_values[0], self.clip_values[1])
        
        output.set_shape(inputs.shape)
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'clip_values': self.clip_values,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))