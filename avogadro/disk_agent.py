# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
from agent import AvogadroAgent
import psutil

from model_params import getModelParams



class AvogadroDiskReadBytesAgent(AvogadroAgent):
  name = "DiskReadBytes"
  datasourceType = "DERIVE"
  minVal = 0
  maxVal = 1000000
  numBuckets = 284
  resolution = max(0.001, (maxVal - minVal) / numBuckets)

  ENCODER_PARAMS = {
    name: {
      "name": name,
      "fieldname": name,
      "resolution": resolution,
      "seed": 42,
      "type": "RandomDistributedScalarEncoder"
    }
  }

  MODEL_PARAMS = getModelParams(ENCODER_PARAMS, name)

  def collect(self):
    return psutil.disk_io_counters().read_bytes



class AvogadroDiskWriteBytesAgent(AvogadroAgent):
  name = "DiskWriteBytes"
  datasourceType = "DERIVE"
  minVal = 0
  maxVal = 1000000
  numBuckets = 284
  resolution = max(0.001, (maxVal - minVal) / numBuckets)

  ENCODER_PARAMS = {
    name: {
      "name": name,
      "fieldname": name,
      "resolution": resolution,
      "seed": 42,
      "type": "RandomDistributedScalarEncoder"
    }
  }

  MODEL_PARAMS = getModelParams(ENCODER_PARAMS, name)

  def collect(self):
    return psutil.disk_io_counters().write_bytes



class AvogadroDiskReadTimeAgent(AvogadroAgent):
  name = "DiskReadTime"
  datasourceType = "DERIVE"
  minVal = 0
  maxVal = 1000
  numBuckets = 284
  resolution = max(0.001, (maxVal - minVal) / numBuckets)

  ENCODER_PARAMS = {
    name: {
      "name": name,
      "fieldname": name,
      "resolution": resolution,
      "seed": 42,
      "type": "RandomDistributedScalarEncoder"
    }
  }

  MODEL_PARAMS = getModelParams(ENCODER_PARAMS, name)

  def collect(self):
    return psutil.disk_io_counters().read_time



class AvogadroDiskWriteTimeAgent(AvogadroAgent):
  name = "DiskWriteTime"
  datasourceType = "DERIVE"
  minVal = 0
  maxVal = 3000
  numBuckets = 284
  resolution = max(0.001, (maxVal - minVal) / numBuckets)

  ENCODER_PARAMS = {
    name: {
      "name": name,
      "fieldname": name,
      "resolution": resolution,
      "seed": 42,
      "type": "RandomDistributedScalarEncoder"
    }
  }

  MODEL_PARAMS = getModelParams(ENCODER_PARAMS, name)

  def collect(self):
    return psutil.disk_io_counters().write_time
