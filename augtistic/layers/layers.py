import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec

import augtistic.random as augr

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomSaturation(Layer):
    """Adjust the contrast of an image or images by a random factor.
    Contrast is adjusted independently for each channel of each image during
    training.
    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The contrast factor will be randomly picked between
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

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomHue(Layer):
    """Adjust the contrast of an image or images by a random factor.
    Contrast is adjusted independently for each channel of each image during
    training.
    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The contrast factor will be randomly picked between
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
        super(RandomHue, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_hue_inputs():
            return tf.image.random_hue(inputs, self.factor, self.seed)

        output = tf_utils.smart_cond(training, random_hue_inputs,
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
        base_config = super(RandomHue, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="Augtistic")
class RandomBrightness(Layer):
    """Adjust the contrast of an image or images by a random factor.
    Contrast is adjusted independently for each channel of each image during
    training.
    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.
    Attributes:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a single
            float, lower = upper. The contrast factor will be randomly picked between
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
        super(RandomBrightness, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_brightness_inputs():
            return tf.image.random_brightness(inputs, 
                                              self.factor, 
                                              self.seed)

        output = tf_utils.smart_cond(training, random_brightness_inputs,
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
        base_config = super(RandomBrightness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        self._rng = make_generator(self.seed)
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

    def __init__(self, name=None, **kwargs):
        self.input_spec = InputSpec(ndim=4)
        super(RandomGrayscale, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_grayscale_inputs():
            return self._color_drop(inputs)

        output = tf_utils.smart_cond(training, random_grayscale_inputs,
                                     lambda: inputs)
        output.set_shape(inputs.shape)
        return output

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
    and `factor`. Uses ref:tfa.image.blend to combine images.
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

    def __init__(self, name=None, **kwargs):
        self.factor = factor
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
        self._rng = augr.generator.get(self.seed)
        super(RandomBlendGrayscale, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_grayscale_inputs():
            return self._color_drop(inputs)

        output = tf_utils.smart_cond(training, random_grayscale_inputs,
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
        base_config = super(RandomBlendGrayscale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _color_blend_drop(self, x):
        blend = self._rng.uniform([], self.lower, self.upper, 
                                  dtype=tf.float32)
        gray = tf.image.rgb_to_grayscale(x)
        gray = tf.tile(gray, [1, 1, 1, 3])
        return tfa.image.blend(gray, x, blend)

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