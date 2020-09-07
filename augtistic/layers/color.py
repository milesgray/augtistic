import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

class RandomHSVinYIQ(Layer):
    """Adjust hue, saturation, value of an RGB image randomly in YIQ color space.
    Equivalent to adjust_yiq_hsv() but uses:
    - delta_h randomly picked in the interval [-max_delta_hue, max_delta_hue], 
    - scale_saturation randomly picked in the interval [lower_saturation, upper_saturation], 
    - scale_value randomly picked in the interval [lower_saturation, upper_saturation].
    Input shape:
        3D tensor with shape:
        `(height, width, channels)`, data_format='channels_last'.
    Output shape:
        3-D float Tensor witj shape:
        `(height, width, channels)`, data_format='channels_last'.
    Attributes:
        image:	RGB image or images. Size of the last dimension must be 3.
        max_delta_hue:	float. Maximum value for the random delta_hue. Passing 0 disables adjusting hue.
        lower_saturation:	float. Lower bound for the random scale_saturation.
        upper_saturation:	float. Upper bound for the random scale_saturation.
        lower_value:	float. Lower bound for the random scale_value.
        upper_value:	float. Upper bound for the random scale_value.
        seed:	An operation-specific seed. It will be used in conjunction with the graph-level seed to determine the real seeds that will be used in this operation. Please see the documentation of set_random_seed for its interaction with the graph-level random seed.
        name:	A name for this operation (optional).
    Raise:
        ValueError	if max_delta, lower_saturation, upper_saturation, 
                    lower_value, or upper_value is invalid.
    """

    def __init__(self, max_delta_hue, lower_saturation, upper_saturation, 
                 lower_value, upper_value, seed=random.randint(0,1000), name=None, **kwargs):
        self.max_delta_hue = max_delta_hue
        self.lower_saturation = lower_saturation
        self.upper_saturation = upper_saturation
        self.lower_value = lower_value
        self.upper_value = upper_value

        if self.lower_saturation < 0. or self.upper_saturation < 0. or self.lower_value < 0.:
            raise ValueError('Cannot have negative values or greater than 1.0,'
                                             ' got {}'.format(factor))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomHSVinYIQ, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_hsv_in_yiq_inputs(x):
            return tfa.image.random_hsv_in_yiq(x,
                                               self.fmax_delta_hue,
                                               self.lower_saturation,
                                               self.upper_saturation,
                                               self.lower_value,
                                               self.upper_value,
                                               self.seed,
                                               self.name
                    )

        output = tf_utils.smart_cond(training, random_hsv_in_yiq_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
                'max_delta_hue': self.max_delta_hue,
                'lower_saturation': self.lower_saturation,
                'upper_saturation': self.upper_saturation,
                'lower_value': self.lower_value,
                'upper_value': self.upper_value,
                'name': self.name,
                'seed': self.seed,
        }
        base_config = super(RandomHSVinYIQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))