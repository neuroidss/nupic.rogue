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
    except (ValueError, TypeError) as e:
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

  (options, args) = parser.parse_args()

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
