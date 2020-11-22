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

        def random_hsv_in_yiq_inputs():
            return tfa.image.random_hsv_in_yiq(inputs,
                                               self.max_delta_hue,
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

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomColorJitter(Layer):
    """Adjust brightness, contrast, saturation and hue of an RGB image randomly.

    Equivalent to calling:
    ```python
        x = tf.image.random_brightness(x, max_delta=max_delta_bright)
        x = tf.image.random_contrast(x, lower=contrast_range[0], upper=contrast_range[1])
        x = tf.image.random_saturation(x, lower=saturation_range[0], upper=saturation_range[1])
        x = tf.image.random_hue(x, max_delta=max_delta_hue)
        x = tf.clip_by_value(x, value_range[0], value_range[1])
    ```

    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
        
    Attributes:
        image:	RGB image or images. Size of the last dimension must be 3.
        max_delta_bright: float. Maximum value for the random delta_hue. Passing 0 disables adjusting hue.
        contrast_factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The contrast factor will be randomly picked between
            [1.0 - lower, 1.0 + upper].
        saturation_factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The saturation factor will be randomly picked between
            [1.0 - lower, 1.0 + upper].
        max_delta_hue:	float. Maximum value for the random delta_hue. Passing 0 disables adjusting hue.
        value_range:	tuple of size 2 representing lower and upper bound on valid image values. 
            Generally one of:
                (0., 1.) or (-1., 1.) or (0, 255) or (-127, 127)
            If only a single value is provided, one of those 4 values will be
            chosen based on some internal rules.
        seed:	An operation-specific seed. It will be used in conjunction with the graph-level seed to determine the real seeds that will be used in this operation. Please see the documentation of set_random_seed for its interaction with the graph-level random seed.
        name:	A name for this operation (optional).
    
    Raise:
        ValueError	if max_delta_bright, contrast_factor, saturation_factor, 
                    max_delta_hue, or value_range is invalid.
    """

    def __init__(self, max_delta_bright, contrast_factor, saturation_factor, 
                 max_delta_hue, value_range, seed=random.randint(0,1000), name=None, **kwargs):
        self.max_delta_bright = self._check_delta(max_delta_bright, name="max_delta_bright")

        self.contrast_factor = self._check_factor(contrast_factor, name="contrast_factor")
        self.contrast_lower = self.contrast_factor[0]
        self.contrast_upper = self.contrast_factor[1]

        self.saturation_factor = self._check_factor(saturation_factor, name="saturation_factor")
        self.saturation_lower = self.saturation_factor[0]
        self.saturation_upper = self.saturation_factor[1]

        self.max_delta_hue = self._check_delta(max_delta_hue, name="max_delta_hue")
        self.value_range = self._check_value_range(value_range)
        self.lower_value = self.value_range[0]
        self.upper_value = self.value_range[1]
        
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomColorJitter, self).__init__(name=name, **kwargs)

    def _check_factor(self, factor, lower_limit=0., upper_limit=1., name="Factor"):
        if isinstance(factor, (tuple, list)):
            lower = factor[0]
            upper = factor[1]
        else:
            lower = upper = factor
        if lower < lower_limit or upper < lower_limit or lower > upper_limit:
            raise ValueError('{} cannot have values less than {} or greater than {},'
                             ' got {}'.format(name, lower_limit, upper_limit, factor))
        return (lower, upper)

    def _check_delta(self, delta, lower_limit=0., upper_limit=0.5, name="Delta"):
        if delta < lower_limit or delta > upper_limit:
            raise ValueError('{} cannot have values less than {} or greater than {},'
                             ' got {}'.format(name, lower_limit, upper_limit, delta))
        return delta

    def _check_value_range(self, value_range, name="value_range"):
        if not isinstance(value_range, tuple) and not isinstance(value_range, list):
            if isinstance(value_range, float):
                if value_range > 0:
                    value_range = (-1., 1.)
                else:
                    value_range = (0.,1.)  
            elif isinstance(value_range, int):
                if value_range > 127:
                    value_range = (0, 255)
                else:
                    value_range = (-127,127)
        if len(value_range) > 2:
            raise ValueError("{} is more than 2 values, should be the lower and upper bound on values!".format(name))

        return value_range

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_color_jitter_inputs():
            x = inputs
            x = tf.image.random_brightness(x, max_delta=self.max_delta_bright, seed=self.seed)
            x = tf.image.random_contrast(x, lower=self.contrast_lower, upper=self.contrast_upper, seed=self.seed)
            x = tf.image.random_saturation(x, lower=self.saturation_lower, upper=self.saturation_upper, seed=self.seed)
            x = tf.image.random_hue(x, max_delta=self.max_delta_hue, seed=self.seed)
            x = tf.clip_by_value(x, self.lower_value, self.upper_value)
            return x

        output = tf_utils.smart_cond(training, random_color_jitter_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
                'max_delta_bright': self.max_delta_bright,
                'contrast_lower': self.contrast_lower,
                'contrast_upper': self.contrast_upper,
                'lower_saturation': self.lower_saturation,
                'upper_saturation': self.upper_saturation,
                'lower_value': self.lower_value,
                'upper_value': self.upper_value,
                'max_delta_hue': self.max_delta_hue,
                'name': self.name,
                'seed': self.seed,
        }
        base_config = super(RandomColorJitter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))