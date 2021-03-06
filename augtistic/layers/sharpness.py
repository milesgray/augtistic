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
class RandomSharpness(Layer):
    """Adjust the sharpness of an image or images by a random factor.
    Uses the TensorFlow Addons ref:tfa.image.sharpness function.
    https://www.tensorflow.org/addons/api_docs/python/tfa/image/sharpness
    
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    
    Attributes:
        probability: A positive float, chance of applying at all
        factor: A positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The sharpness factor will be randomly picked between
            [lower, upper].
        blend_factor: A positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound of the amount to blend the result. 
            When represented as a single float, blend_lower = blend_upper. 
            The blend factor will be randomly picked between [blend_lower, blend_upper].            
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    
    Raise:
        ValueError: if lower bound or upper bound is negative, or if upper bound is
                    smaller than lower bound
        ValueError: if blend lower bound or upper bound is negative, 
                    or if blend upper bound is smaller than lower bound
    """

    def __init__(self, probability, factor, blend_factor=(0.1,0.7), seed=random.randint(0,1000), name=None, **kwargs):
        self.probability = probability
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor
        if self.lower < 0. or self.upper < 0.:
            raise ValueError('Factor cannot have negative values,'
                             ' got {}'.format(factor))
        if self.upper < self.lower:
            raise ValueError('Factor first value must be smaller than second,'
                             ' got {}'.format(factor))
        self.blend_factor = blend_factor
        if isinstance(blend_factor, (tuple, list)):
            self.blend_lower = blend_factor[0]
            self.blend_upper = blend_factor[1]
        else:
            self.blend_lower = self.blend_upper = blend_factor
        if self.blend_lower < 0. or self.blend_upper < 0.:
            raise ValueError('Blend Factor cannot have negative values,'
                             ' got {}'.format(blend_factor))
        if self.blend_upper < self.blend_lower:
            raise ValueError('Blend Factor first value must be smaller than second,'
                             ' got {}'.format(blend_factor))
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        self._rng = augr.get(self.seed)
        super(RandomSharpness, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_sharpness_inputs():
            return self._random_apply(self._apply_sharp, inputs, self.probability)

        output = tf_utils.smart_cond(training, random_sharpness_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(self._rng.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)
        
    def _apply_sharp(self, inputs):
        factor = self._rng.uniform(shape=[], 
                                   minval=self.lower, 
                                   maxval=self.upper, dtype=tf.float32)
        blend = self._rng.uniform(shape=[], 
                                   minval=self.blend_lower, 
                                   maxval=self.blend_upper, dtype=tf.float32)
        sharp_inputs = tfa.image.sharpness(inputs, 
                                   factor=factor)
        return tfa.image.blend(sharp_inputs, inputs, blend)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'factor': self.factor,
            'seed': self.seed
        }
        base_config = super(RandomSharpness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
