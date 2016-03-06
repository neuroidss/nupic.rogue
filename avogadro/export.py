#!/usr/bin/env python
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

""" Utility for fetching metric data from local database, and forwarding to
a Grok server via Custom Metrics Socket API.
"""

import csv
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
      start = str(int(time.mktime((datetime.datetime.now() -
                                   datetime.timedelta(days=14)).timetuple())))

    fetched = metric.fetch(prefix=options.prefix, start=start)

    for (ts, value) in fetched:
      try:
        value = float(value)
      except (ValueError, TypeError):
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

  (options, _args) = parser.parse_args()

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
