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
import time
import gevent
from gevent import Greenlet
from rrdtool import RRDToolClient



DEFAULT_INTERVAL = 60



class AvogadroAgent(RRDToolClient, Greenlet):
  """ Avogadro Agent base class implementation.
  """


  @property
  def name(self):
    return self.__class__.__name__


  def __init__(self, interval=DEFAULT_INTERVAL, options=None):
    super(AvogadroAgent, self).__init__(options=options)
    self.interval = options.interval or interval


  def __repr__(self):
    return self.name


  def _run(self):
    while True:
      value = self.collect() # collect() implemented in subclass
      if value is None:
        continue # skip non-values
      ts = time.time()
      print self, ts, value
      super(AvogadroAgent, self).store(value, ts=ts) # store() implemented in
                                                     # super()
      gevent.sleep(self.interval)


  @classmethod
  def addParserOptions(cls, parser):
    super(AvogadroAgent, cls).addParserOptions(parser)
    parser.add_option("--interval",
                       default=DEFAULT_INTERVAL,
                       dest="interval",
                       help="Interval, in seconds, for metric collection",
                       metavar="SECONDS",
                       type="int")



  def collect(self):
    raise NotImplementedError("collect() not implemented in subclass")


