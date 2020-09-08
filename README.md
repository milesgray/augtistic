# AugTistic

**AugTistic** is a repository that expands on the TensorFlow's [Keras Preprocessing layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing) introduced in TensorFlow 2.3.0 under the `tf.keras.layers.experimental.preprocessing` namespace. This repository structure is based on TensorFlow Addons and the layers themselves are based on the 2.3.0 implementations of the `RandomContrast` Preprocessing layer, which is likely to change in future versions.  As it is now, however, these layers will most likely continue to work as they are merely subclassed from the standard Keras `Layer` base class.

These layers were made with the recent Vision-based **Contrastive Learning** trend in mind, in particular the instance discrimination strategy that is proving to be a core component to successful contrastive pipelines. Of course, these layers can be used in any situation where the data augmentation should be built into a Model instead of external code - or if you just get fixated on the layer abstraction and need as much functionality as possible within it to avoid throwing a tantrum.

Note: *This is currently in **BETA** status as all bugs are ironed out through usage*

## Available Augmentations

* [Random Saturation](augtistic/layers/saturation.py)
* [Random Hue](augtistic/layers/hue.py)
* [Random Brightness](augtistic/layers/brightness.py)
* **Grayscale**
  * [Random Grayscale](augtistic/layers/grayscale.py)
  * [Random Blending Grayscale](augtistic/layers/grayscale.py)
* [Random Cutout](augtistic/layers/cutout.py)
* [Random Warp](augtistic/layers/warp.py)
* [Random Sharpness](augtistic/layers/sharpness.py)
* **Blur**
  * [Random Gaussian 2D Blur](augtistic/layers/blur.py)
  * [Random Mean Filter 2D](augtistic/layers/blur.py)
  * [Random Median Filter 2D](augtistic/layers/blur.py)
* [Random HSV in YIQ](augtistic/layers/color.py)

## Installation

AugTistic will soon be made available on PyPI, once I am confident that all of the bugs are out. In the meantime, to install the package you must clone this repo and manually install it using the following commands:

```shell
git clone https://github.com/milesgray/augtistic.git
cd augtistic
python setup.py install
```

Alternatively, just simply copy and paste the `layers` folder into your codebase and rename it `augtistic`

## Usage

To use the layers, import the package and simply add the new layers to your model just like any other Keras layer. One way is to include the augmentation layers directly into the full model:

```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as tfpp
import augtistic as tfaug

IMAGE_SHAPE = (224,224,3)
NUM_CLASSES = 5

model = Sequential([
  tfpp.Rescaling(1./255, input_shape=IMAGE_SHAPE),
  tfaug.RandomGrayscale(),
  tfaug.RandomCutout(20),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(NUM_CLASSES)
])
```

Another pattern that can be used is to make a stand-alone model consisting of only augmentation layers that data is passed through before being sent into the main model in a custom training loop.

```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as tfpp
import augtistic as tfaug

IMAGE_SHAPE = (224,224,3)

aug_input = layers.Input(IMAGE_SHAPE)
aug_output = tfpp.RandomContrast(0.2)(aug_input)
aug_output = tfaug.RandomSaturation(0.2)(aug_output)
aug_output = tfaug.RandomBrightness(0.2)(aug_output)
aug_output = tfaug.RandomHue(0.2)(aug_output)
aug_output = tfaug.RandomCutout(20)(aug_output)
aug_output = tfaug.RandomCutout(4)(aug_output)
aug_output = tfaug.RandomCutout(28, )(aug_output)
aug_output = tfpp.RandomZoom((-0.25,0.2), width_factor=(-0.25,0.2))(aug_output)
aug_output = tfpp.RandomTranslation((-0.1, 0.1), (-0.15, 0.15))(aug_output)
aug_model = keras.Model(aug_input, aug_output)

base_model = keras.applications.ResNetV2(input_shape=IMAGE_SHAPE,
                                         include_top=False,
                                         weights='imagenet')
# get data from a previously created tensorflow dataset, but could be any numpy array of images
image_batch, label_batch = next(iter(train_dataset))
# applies the transformations to the batch and results in a numpy array
# alternatively, could pass the image batch directly into the model to get a tensor array
image_batch = aug_model.predict(image_batch)
feature_batch = base_model(image_batch)
```

For a more in-depth look at the Keras Data Augmentation offerings for images, refer to the official [tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).

*These layers can also be used in dataset pipelines!*

## License

[MIT](LICENSE)
