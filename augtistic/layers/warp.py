import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomDenseImageWarp(Layer):
    """This operation is for non-linear warp of any image specified by the flow field of the offset vector (here used random values for example).
    Uses the TensorFlow Addons ref:tfa.image.dense_image_warp function.
    https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The sharpness factor will be randomly picked between
            [lower, upper].
        seed: Integer. Used to create a random seed.
        name: A string, the name of the layer.
    Raise:
        ValueError: if lower bound or upper bound is negative, or if upper bound is
                    smaller than lower bound
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
        self._rng = make_generator(self.seed)
        super(RandomDenseImageWarp, self).__init__(name=name, **kwargs)
    
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_warp_inputs():
            return self._random_apply(self._apply_warp, inputs, self.probability)

        output = tf_utils.smart_cond(training, random_warp_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(self._rng.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)
        
    def _apply_warp(self, inputs):
        blend = self._rng.uniform(shape=[], 
                                   minval=self.blend_lower, 
                                   maxval=self.blend_upper, dtype=tf.float32)        
        flow_shape = [1, self.height, self.width, 2]
        init_flows = self._rng.uniform(shape=flow_shape, 
                                   minval=self.lower, 
                                   maxval=self.upper, dtype=tf.float32)
        init_flows = init_flows * 2.0
        dense_img_warp = tfa.image.dense_image_warp(inputs, init_flows)
        return tfa.image.blend(inputs, dense_img_warp, blend)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'factor': self.factor,
            'seed': self.seed
        }
        base_config = super(RandomDenseImageWarp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))