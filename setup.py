# Copyright 2020 Miles Gray
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==============================================================================
"""AugTistic.

AugTistic is a repository that expands on the Keras Preprocessing layers introduced in 
TensorFlow 2.3.0 under the `tf.keras.layers.experimental.preprocessing`
namespace. This repository structure is based on TensorFlow Addons and the layers
themselves are based on the 2.3.0 implementations of the Preprocessing layers, which
is likely to change.  As it is now, however, these layers will most likely continue to
work as they are merely subclassed from the standard Keras `Layer` base class. 
"""

import os
from pathlib import Path
import sys

from datetime import datetime
from setuptools import find_packages
from setuptools import setup

version_path = os.path.join(os.path.dirname(__file__), 'augtistic')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

DOCLINES = __doc__.split("\n")

setup(
    name='augtistic',
    version=__version__,
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="Miles Gray",
    author_email="miles@resatiate.com",
    url="https://github.com/milesgray/augtistic",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    keywords="tensorflow image augmentation machine learning computer vision",
)