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
""" Avogadro Agents save metric data to local Round Robin Database (RRD)
"""
from itertools import imap
import os
from StringIO import StringIO
import subprocess
import time



DEFAULT_HEARTBEAT = 600
DEFAULT_RESET = False



class RRDToolClient(object):
  """ Helper utility for interfacing with rrdtool """

  # RRDTool default DS: arguments for `rrdcreate`
  datasourceType = "GAUGE"
  minVal = 0
  maxVal = "U"
  heartbeat = DEFAULT_HEARTBEAT


  def __init__(self, options):
    """
    :param options: CLI options
    """
    super(RRDToolClient, self).__init__()
    self.options = options
    self.heartbeat = options.heartbeat or self.heartbeat
    self._rrdcreate(prefix=options.prefix,
                    reset=options.reset,
                    step=options.interval)


  @classmethod
  def createParams(cls):
    # Subclasses may override name, datasourceType, min, or max
    ds = [":".join(["DS",
                    cls.name[:19], # A ds-name must be 1 to 19 characters
                                    # long in the characters [a-zA-Z0-9_]
                                    # http://oss.oetiker.ch/rrdtool/doc/rrdcreate.en.html
                    cls.datasourceType,
                    str(cls.heartbeat),
                    str(cls.minVal),
                    str(cls.maxVal)])]

    rra = ["RRA:AVERAGE:0.5:1:4032"] # 14 days - 5-minute resolution

    return (ds, rra)


  @classmethod
  def addParserOptions(cls, parser):
    parser.add_option("--prefix",
                      default=None,
                      dest="prefix",
                      help="Location of databases",
                      metavar="PATH")
    parser.add_option("--hearbeat",
                      default=DEFAULT_HEARTBEAT,
                      dest="heartbeat",
                      help="RRDTool heartbeat",
                      metavar="SECONDS",
                      type="int")
    parser.add_option("--reset",
                      default=DEFAULT_RESET,
                      dest="reset",
                      help="Reset databases",
                      action="store_true")


  def store(self, value, ts=None):
    """ Store metric value at time.

    :param value: Metric value
    :param ts: Timestamp
    """
    self._rrdupdate(value, prefix=self.options.prefix, n=ts)


  def _rrdupdate(self, value, n=None, prefix=None):
    """ `rrdtool update`

    :param value: Metric value
    :param n: Timestamp
    :param prefix: Location of databases
    """
    n = n or "N"
    filename = []
    if prefix:
      filename.append(os.path.normpath(prefix))

    filename.append(".".join([self.name, "rrd"]))

    filepath = os.path.join(*filename)

    subprocess.check_call(["rrdtool",
                           "update",
                           filepath,
                           "%s:%d" % (n, value)])


  @classmethod
  def _rrdcreate(cls, prefix=None, start=None, step=300, reset=DEFAULT_RESET):
    """ `rrdtool create`

    :param prefix: Location of databases
    :param start: Start time (unix timestamp)
    :param step: Interval between metric data points
    :param reset: Delete previously created databases
    """

    start = start or int(time.time())

    filename = []
    if prefix:
      filename.append(os.path.normpath(prefix))

    filename.append(".".join([cls.name, "rrd"]))

    filepath = os.path.join(*filename)

    if reset and os.path.exists(filepath):
      os.unlink(filepath)

    try:
      os.makedirs(os.path.dirname(filepath))
    except os.error:
      if not os.path.isdir(os.path.dirname(filepath)):
        raise
      else:
        pass # suppress error.  desired path already exists

    (ds, rra) = cls.createParams()

    subprocess.check_call(["rrdtool",
                           "create",
                           filepath,
                           "--step", "%d" % step,
                           "--start", "%d" % start] + ds + rra)


  @classmethod
  def fetch(cls, prefix=None, start=None):
    """ Fetches and returns metric data from local database.

    :param prefix: Location (on disk) of databases
    :param start: Optional starting point in time
    :returns: metric iterator
    """
    fetched = cls._rrdfetch(prefix=prefix, start=start)

    def lineProcessor(line):
      components = line.split(":")
      if len(components) == 2:
        return [component.strip() for component in line.split(":")]
      else:
        return [None, None]

    return imap(lineProcessor, StringIO(fetched))


  @classmethod
  def _rrdfetch(cls, prefix=None, start=None):
    """ `rrdtool fetch` """
    filename = []
    if prefix:
      filename.append(os.path.normpath(prefix))

    filename.append(".".join([cls.name, "rrd"]))

    filepath = os.path.join(*filename)

    args = []
    if start is not None:
      args.extend(["--start", start])

    return subprocess.check_output(["rrdtool",
                                    "fetch",
                                    filepath,
                                    "AVERAGE"] + args)
