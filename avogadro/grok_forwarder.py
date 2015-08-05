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
a Grok server via Custom Metrics Socket API.
"""
import math
import os
from optparse import OptionParser

from grokcli.api import GrokSession

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



def _fetchAndForward(sock, metric, options):
  """ Fetch metrics from local database, forward to Grok server custom metrics
  API.

  :param sock: Open socket connection to Grok server
  :param metric: AvogadroAgent metric class
  :param options: CLI Options
  """

  target = os.path.join(options.prefix, metric.name + ".pos")

  if os.path.exists(target):
    mode = "r+"
  else:
    mode = "w"

  with open(target, mode) as fp:
    if mode == "r+":
      start = fp.read()
    else:
      start = None

    fetched = metric.fetch(prefix=options.prefix, start=start)

    for (ts, value) in fetched:
      try:
        value = float(value)
      except (ValueError, TypeError):
        continue

      if not math.isnan(value):
        sock.sendall("%s %s %s\n" % (metric.name, value, ts))
        start = ts

    else:
      fp.seek(0)
      fp.write(str(start))
      fp.truncate()



def main():
  """ Main entry point for Grok Custom Metric forwarder """

  parser = OptionParser()
  parser.add_option("--server",
                    dest="server",
                    help="Grok server")

  AvogadroAgent.addParserOptions(parser)

  (options, _args) = parser.parse_args()

  grok = GrokSession(server=options.server)

  with grok.connect() as sock:
    _fetchAndForward(sock, AvogadroCPUTimesAgent, options)
    _fetchAndForward(sock, AvogadroMemoryAgent, options)
    _fetchAndForward(sock, AvogadroDiskReadBytesAgent, options)
    _fetchAndForward(sock, AvogadroDiskWriteBytesAgent, options)
    _fetchAndForward(sock, AvogadroDiskReadTimeAgent, options)
    _fetchAndForward(sock, AvogadroDiskWriteTimeAgent, options)
    _fetchAndForward(sock, AvogadroNetworkBytesSentAgent, options)
    _fetchAndForward(sock, AvogadroNetworkBytesReceivedAgent, options)
    _fetchAndForward(sock, AvogadroKeyCountAgent, options)
    _fetchAndForward(sock, AvogadroKeyDownDownAgent, options)
    _fetchAndForward(sock, AvogadroKeyUpDownAgent, options)
    _fetchAndForward(sock, AvogadroKeyHoldAgent, options)



if __name__ == "__main__":
  main()
