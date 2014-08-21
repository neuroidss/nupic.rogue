#------------------------------------------------------------------------------
# Copyright 2013-2014 Numenta Inc.
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
#------------------------------------------------------------------------------
"""
  Numenta Rogue
  =============

  Metric collection agent for the Numenta Rogue showcase application.

  Installation notes
  ------------------

  This file (setup.py) is provided to support installation using the native
  python setuptools-based ecosystem, including PyPi, `easy_install` and `pip`.

  Disclaimer:  Your specific environment _may_ require additional arguments to
  pip, setup.py and easy_install such as `--root`, `--install-option`,
  `--script-dir`, `--script-dir`, or you may use `sudo` to install at the system
  level.

  Building source distribution for release
  ----------------------------------------

  The source distribution package is built using the `sdist build` sub-command:

      python setup.py sdist build

  Resulting in the creation of dist/numenta-rogue-X.Y.tar.gz, which will be
  uploaded to PyPi (or another distribution channel).  The numenta-rogue
  package can be installed from the tarball directly using a number of
  approaches:

      pip install numenta-rogue-X.Y.tar.gz
      easy_install numenta-rogue-X.Y.tar.gz

  Or, by using setup.py:

      tar xzvf numenta-rogue-X.Y.tar.gz
      cd numenta-rogue-X.Y.tar.gz
      python setup.py install

  Once uploaded to PyPi, numenta-rogue can be installed by name:

      pip install numenta-rogue
      easy_install numenta-rogue

  Alternate installation by `pip wheel`
  -------------------------------------

  Recently, pip has added a new binary distribution format "wheel", which
  simplifies the process somewhat.

  To create a wheel:

      pip wheel .

  Resulting in the creation of wheelhouse/numenta-rogue-X.Y-py27-none-any.whl
  along with a few other .whl files related to dependencies.

  To install from cached wheel:

      pip install --use-wheel --no-index --find-links=wheelhouse/ wheelhouse/numenta-rogue-X.Y-py27-none-any.whl

  Or, from PyPi, assuming the wheels have been uploaded to PyPi:

      pip install --use-wheel numenta-rogue


  Uploading to PyPi
  -----------------

  Build and upload source, egg, and wheel distributions to PyPi:

      python setup.py sdist bdist_wheel bdist_egg upload
"""
from setuptools import find_packages, setup



requirements = map(str.strip, open("requirements.txt").readlines())

version = {}
execfile("avogadro/__version__.py", {}, version)

setup(
  name = "numenta-rogue",
  description = (
    "Metric collection agent for Numenta Rogue showcase application"),
  url = "https://github.com/numenta/nupic.rogue",
  classifiers = [
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 2",
    "Topic :: Utilities"],
  keywords = "grok, numenta, anomaly detection, monitoring, nupic",
  author = "Austin Marshall",
  author_email = "amarshall@numenta.com",
  packages = find_packages(),
  entry_points = (
    {"console_scripts": ["rogue-agent = avogadro:main",
                         "rogue-export = avogadro.export:main",
                         "rogue-forward = avogadro.grok_forwarder:main",
                         "rogue-keycounter = avogadro.keys:main"]}),
  install_requires = requirements,
  extras_require = {"docs": ["sphinx"]},
  version = version["__version__"]
)
