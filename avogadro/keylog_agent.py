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
import os, csv

from model_params import getModelParams



class AvogadroKeyCountAgent(AvogadroAgent):
  name = "KeyCount"
  minVal = 0.0
  maxVal = 1000.0
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
    inPath = os.path.dirname(os.path.realpath(__file__)) + "/keys.temp"
    if os.path.isfile(inPath):
      with open(inPath, "rb") as inputFile:
        csvreader = csv.reader(inputFile)
        for row in csvreader:
          data = int(row[0])
          return data

class AvogadroKeyDownDownAgent(AvogadroAgent):
  name = "KeyDownDown"
  minVal = 0.0
  maxVal = 300.0
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
    inPath = os.path.dirname(os.path.realpath(__file__)) + "/keys.temp"
    if os.path.isfile(inPath):
      with open(inPath, "rb") as inputFile:
        csvreader = csv.reader(inputFile)
        for row in csvreader:
          data = float(row[1])
          return data

class AvogadroKeyUpDownAgent(AvogadroAgent):
  name = "KeyUpDown"
  minVal = 0.0
  maxVal = 300.0
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
    inPath = os.path.dirname(os.path.realpath(__file__)) + "/keys.temp"
    if os.path.isfile(inPath):
      with open(inPath, "rb") as inputFile:
        csvreader = csv.reader(inputFile)
        for row in csvreader:
          data = float(row[2])
          return data

class AvogadroKeyHoldAgent(AvogadroAgent):
  name = "KeyHold"
  minVal = 0.0
  maxVal = 2.0
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
    inPath = os.path.dirname(os.path.realpath(__file__)) + "/keys.temp"
    if os.path.isfile(inPath):
      with open(inPath, "rb") as inputFile:
        csvreader = csv.reader(inputFile)
        for row in csvreader:
          data = float(row[3])
          return data
