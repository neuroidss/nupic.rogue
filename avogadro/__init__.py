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
import gevent
import sys
from optparse import OptionParser

import __version__

# Metric collection agents
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



def main():
  """ Main Avogadro entrypoint.  Runs all metric collection agents. """

  parser = OptionParser(version=__version__.__version__)

  AvogadroAgent.addParserOptions(parser)

  try:
    (options, args) = parser.parse_args(sys.argv)

    gevent.joinall([AvogadroCPUTimesAgent.spawn(options=options),
                    AvogadroMemoryAgent.spawn(options=options),
                    AvogadroDiskReadBytesAgent.spawn(options=options),
                    AvogadroDiskWriteBytesAgent.spawn(options=options),
                    AvogadroDiskReadTimeAgent.spawn(options=options),
                    AvogadroDiskWriteTimeAgent.spawn(options=options),
                    AvogadroNetworkBytesSentAgent.spawn(options=options),
                    AvogadroNetworkBytesReceivedAgent.spawn(options=options),
                    AvogadroKeyCountAgent.spawn(options=options),
                    AvogadroKeyDownDownAgent.spawn(options=options),
                    AvogadroKeyUpDownAgent.spawn(options=options),
                    AvogadroKeyHoldAgent.spawn(options=options)])

  except IndexError:
    parser.print_help(sys.stderr)
    sys.exit()
