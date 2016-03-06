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
