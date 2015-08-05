#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

""" Utility for fetching metric data from local database, and forwarding to
NuPIC on a local machine.
"""

import csv
import datetime
import math
import os
from optparse import OptionParser

from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood
from nupic.frameworks.opf.modelfactory import ModelFactory

from agent import AvogadroAgent
from cpu_agent import AvogadroCPUTimesAgent
from disk_agent import (AvogadroDiskReadBytesAgent,
                        AvogadroDiskWriteBytesAgent,
                        AvogadroDiskReadTimeAgent,
                        AvogadroDiskWriteTimeAgent)
from memory_agent import AvogadroMemoryAgent
from network_agent import (AvogadroNetworkBytesSentAgent,
                           AvogadroNetworkBytesReceivedAgent)
from keylog_agent import (AvogadroKeyCountAgent,
                          AvogadroKeyDownDownAgent,
                          AvogadroKeyUpDownAgent,
                          AvogadroKeyHoldAgent)



def createModel(metric):
  """
    Fetch the model params for the specific metric and create a new HTM model

    :param metric: AvogadroAgent metric class

    :returns: An HTM Model
  """
  return ModelFactory.create(metric.MODEL_PARAMS)



def runAvogadroAnomaly(metric, options):
  """
  Create a new HTM Model, fetch the data from the local DB, process it in NuPIC,
  and save the results to a new CSV output file.

  :param metric: AvogadroAgent metric class
  :param options: CLI Options
  """
  model = createModel(metric)
  model.enableInference({"predictedField": metric.name})

  fetched = metric.fetch(prefix=options.prefix, start=None)

  resultFile = open(os.path.join(options.prefix, metric.name + "-result.csv"),
                    "wb")
  csvWriter = csv.writer(resultFile)
  csvWriter.writerow(["timestamp", metric.name, "raw_anomaly_score",
                      "anomaly_likelihood", "color"])

  headers = ("timestamp", metric.name)

  anomalyLikelihood = AnomalyLikelihood()

  for (ts, value) in fetched:
    try:
      value = float(value)
    except (ValueError, TypeError):
      continue

    if not math.isnan(value):
      modelInput = dict(zip(headers, (ts, value)))
      modelInput[metric.name] = float(value)
      modelInput["timestamp"] = datetime.datetime.fromtimestamp(
        float(modelInput["timestamp"]))
      result = model.run(modelInput)
      anomalyScore = result.inferences["anomalyScore"]

      likelihood = anomalyLikelihood.anomalyProbability(
        modelInput[metric.name], anomalyScore, modelInput["timestamp"])
      logLikelihood = anomalyLikelihood.computeLogLikelihood(likelihood)

      if logLikelihood > .5:
        color = "red"
      elif logLikelihood > .4 and logLikelihood <= .5:
        color = "yellow"
      else:
        color = "green"

      csvWriter.writerow([modelInput["timestamp"], float(value),
                          anomalyScore, logLikelihood, color])

  else:
    resultFile.flush()



def main():
  """ Main entry point for NuPIC Metric forwarder """

  parser = OptionParser()

  AvogadroAgent.addParserOptions(parser)

  (options, _args) = parser.parse_args()

  runAvogadroAnomaly(AvogadroCPUTimesAgent, options)
  runAvogadroAnomaly(AvogadroMemoryAgent, options)
  runAvogadroAnomaly(AvogadroDiskReadBytesAgent, options)
  runAvogadroAnomaly(AvogadroDiskWriteBytesAgent, options)
  runAvogadroAnomaly(AvogadroDiskReadTimeAgent, options)
  runAvogadroAnomaly(AvogadroDiskWriteTimeAgent, options)
  runAvogadroAnomaly(AvogadroNetworkBytesSentAgent, options)
  runAvogadroAnomaly(AvogadroNetworkBytesReceivedAgent, options)
  runAvogadroAnomaly(AvogadroKeyCountAgent, options)
  runAvogadroAnomaly(AvogadroKeyDownDownAgent, options)
  runAvogadroAnomaly(AvogadroKeyUpDownAgent, options)
  runAvogadroAnomaly(AvogadroKeyHoldAgent, options)


if __name__ == "__main__":
  main()
