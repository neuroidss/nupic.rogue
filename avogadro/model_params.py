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
import copy



MODEL_PARAMS = {
  "aggregationInfo": {
    "seconds": 0,
    "fields": [],
    "months": 0,
    "days": 0,
    "years": 0,
    "hours": 0,
    "microseconds": 0,
    "weeks": 0,
    "minutes": 0,
    "milliseconds": 0
  },
  "model": "CLA",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "sensorParams": {
      "verbosity": 0,
      "encoders": {
        "timestamp_dayOfWeek": None,
        "timestamp_timeOfDay": {
          "type": "DateEncoder",
          "timeOfDay": [21, 5.4864773611134598],
          "fieldname": "timestamp",
          "name": "timestamp"
        },
        "timestamp_weekend": None
      },
      "sensorAutoReset": None
    },
    "spParams": {
      "spatialImp": "cpp",
      "columnCount": 2048,
      "synPermInactiveDec": 0.00065,
      "inputWidth": 0,
      "spVerbosity": 0,
      "synPermActiveInc": 0.001,
      "synPermConnected": 0.1,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "globalInhibition": 1,
      "maxBoost": 1.0
    },
    "trainSPNetOnlyIfRequested": False,
    "clParams": {
      "alpha": 0.0068717199878650798,
      "regionName": "CLAClassifierRegion",
      "steps": "1",
      "clVerbosity": 0
    },
    "tpParams": {
      "columnCount": 2048,
      "activationThreshold": 13,
      "pamLength": 1,
      "cellsPerColumn": 32,
      "permanenceInc": 0.1,
      "minThreshold": 11,
      "verbosity": 0,
      "maxSynapsesPerSegment": 32,
      "outputType": "normal",
      "globalDecay": 0.0,
      "initialPerm": 0.21,
      "permanenceDec": 0.1,
      "seed": 1960,
      "maxAge": 0,
      "newSynapseCount": 20,
      "maxSegmentsPerCell": 128,
      "temporalImp": "cpp",
      "inputWidth": 2048
    },
    "anomalyParams": {
      "anomalyCacheRecords": None,
      "autoDetectThreshold": None,
      "autoDetectWaitRecords": 5030
    },
    "spEnable": True,
    "inferenceType": "TemporalAnomaly",
    "tpEnable": True,
    "clEnable": False
  }
}



def getModelParams(encoderParams, predictedField):
  """
    Creates a model params dict that includes the encoder params for the
    specific model.

    :param encoderParams: A dict containing the encoder parameters for the
    specified predicted field. For example:
      {
        u"CPUPercent": {
          u"name": u"CPUPercent",
          "fieldname": u"CPUPercent",
          "resolution": 0.3521126761,
          "seed": 42,
          "type": "RandomDistributedScalarEncoder"
        }
      }

    NOTE: The fieldname, name and parent value must all be the same (e.g.,
      CPUPercent)

    :param predictedField: A `string` representing the name of the
    predictedField. This should match exactly the `fieldname` in the encoder
    params

    :returns: A `dict` with all of the relevant model parameters
    :rtype: dict
  """
  thisModel = copy.deepcopy(MODEL_PARAMS)
  thisModel["modelParams"]["sensorParams"]["encoders"][predictedField] = (
    encoderParams[predictedField])
  return thisModel
