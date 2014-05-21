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
import os
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


