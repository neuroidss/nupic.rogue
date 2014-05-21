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
      except (ValueError, TypeError) as e:
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

  (options, args) = parser.parse_args()

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
