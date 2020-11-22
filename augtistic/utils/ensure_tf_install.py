# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Ensure the TensorFlow version is in the right range. This
# needs to happen before anything else, since the imports below will try to
# import TensorFlow, too.

from distutils.version import LooseVersion
import warnings

import tensorflow as tf
import tensorflow_addons as tfa

MIN_TF_VERSION = "2.3.0"
MIN_TFA_VERSION = ""

def check_tf_version():
    """Warn the user if the version of TensorFlow used is not supported.
    This is a check for the new Preprocessing type of layer introduced in TensorFlow 2.3,
    though it is not explicitly referenced.
    """

    min_tf_version = LooseVersion(MIN_TF_VERSION)

    if min_tf_version <= LooseVersion(tf.__version__):
        return

    warnings.warn(
        "Augtistic supports using Keras Layer base class that is in most Tensorflow versions, "
        "but the Preprocessing concept is only in versions above or equal to {}.\n "
        "The versions of TensorFlow you are currently using is {} and is not "
        "supported. \n"
        "Some things might work, some things might not.\n"
        "If you were to encounter a bug, do not file an issue.\n"
        "If you want to make sure you're using a tested and supported configuration, "
        "change the TensorFlow version. See more details at: \n"
        "https://github.com/milesgray/augtistic".format(
            MIN_TF_VERSION, tf.__version__
        ),
        UserWarning,
    )