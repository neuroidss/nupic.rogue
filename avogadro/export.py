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
a Grok server via Custom Metrics Socket API.
"""
import csv
import math
import os
from optparse import OptionParser
import time
import datetime

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



def _fetchAndForward(metric, options, _cache={}):
  """ Fetch metrics from local database, export to CSV file

  :param metric: AvogadroAgent metric class
  :param options: CLI Options
  """

  target = os.path.join(options.prefix, metric.name + "-csv.pos")

  if os.path.exists(target):
    mode = "r+"
  else:
    mode = "w+"

  exportFilename = os.path.join(options.prefix, metric.name + ".csv")

  if exportFilename in _cache:
    (csvout, exportFile) = _cache[exportFilename]
  else:

    if os.path.exists(target):
      exportMode = "a+"
    else:
      exportMode = "w+"

    exportFile = open(exportFilename, exportMode)
    csvout = csv.writer(exportFile)
    _cache[exportFilename] = (csvout, exportFile)

  with open(target, mode) as fp:
    if mode == "r+":
      start = fp.read()
    else:
      start = str(int(time.mktime((datetime.datetime.now()-datetime.timedelta(days=14)).timetuple())))

    fetched = metric.fetch(prefix=options.prefix, start=start)

    for (ts, value) in fetched:
      try:
        value = float(value)
      except (ValueError, TypeError) as e:
        continue

      csvout.writerow((metric.name, value, ts))
      start = ts

    else:
      exportFile.flush()
      fp.seek(0)
      fp.write(str(start))
      fp.truncate()



def main():
  """ Main entry point for Grok Custom Metric exporter """

  parser = OptionParser()

  AvogadroAgent.addParserOptions(parser)

  (options, args) = parser.parse_args()

  _fetchAndForward(AvogadroCPUTimesAgent, options)
  _fetchAndForward(AvogadroMemoryAgent, options)
  _fetchAndForward(AvogadroDiskReadBytesAgent, options)
  _fetchAndForward(AvogadroDiskWriteBytesAgent, options)
  _fetchAndForward(AvogadroDiskReadTimeAgent, options)
  _fetchAndForward(AvogadroDiskWriteTimeAgent, options)
  _fetchAndForward(AvogadroNetworkBytesSentAgent, options)
  _fetchAndForward(AvogadroNetworkBytesReceivedAgent, options)
  _fetchAndForward(AvogadroKeyCountAgent, options)
  _fetchAndForward(AvogadroKeyDownDownAgent, options)
  _fetchAndForward(AvogadroKeyUpDownAgent, options)
  _fetchAndForward(AvogadroKeyHoldAgent, options)



if __name__ == "__main__":
  main()
