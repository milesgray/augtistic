import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomGrayscale(Layer):
    """Convert image or images to grayscale image and restore original 3 channel
    shape.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, probability, seed=random.randint(0,1000), name=None, **kwargs):
        self.probability = probability
        self.seed = seed        
        self.input_spec = InputSpec(ndim=4)
        self._rng = make_generator(self.seed)
        super(RandomGrayscale, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_grayscale_inputs():
            return self._random_apply(self._color_drop, inputs, self.probability)

        output = tf_utils.smart_cond(training, random_grayscale_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(self._rng.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)   
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(RandomGrayscale, self).get_config()

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomBlendGrayscale(Layer):
    """Convert image or images to grayscale image and restore original 3 channel
    shape and then blend with original image at a random amount between 0 
    and `factor`.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = 0. The gray factor will be randomly picked between
            [lower, upper].
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is
            negative.
    """

    def __init__(self, probability, factor=None, seed=random.randint(0,1000), name=None, **kwargs):
        self.probability = probability
        self.factor = factor if factor else probability
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = 0.0
            self.upper = factor
        if self.lower < 0. or self.upper < 0. or self.lower > 1.:
            raise ValueError('Factor cannot have negative values or greater than 1.0,'
                             ' got {}'.format(factor))
        self.seed = seed        
        self.input_spec = InputSpec(ndim=4)
        self._rng = make_generator(self.seed)
        super(RandomBlendGrayscale, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_grayscale_inputs():
            return self._random_apply(self._color_blend_drop, inputs, self.probability)

        output = tf_utils.smart_cond(training, random_grayscale_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

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
            'factor': self.factor,
            'seed': self.seed,
        }
        base_config = super(RandomBlendGrayscale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _color_blend_drop(self, x):
        blend = self._rng.uniform([], self.lower, self.upper, 
                                  dtype=tf.float32)
        gray = tf.image.rgb_to_grayscale(x)
        gray = tf.tile(gray, [1, 1, 1, 3])
        return tfa.image.blend(gray, x, blend)