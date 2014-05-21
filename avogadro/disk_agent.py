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
from agent import AvogadroAgent
import psutil



class AvogadroDiskReadBytesAgent(AvogadroAgent):
  name = "DiskReadBytes"
  datasourceType = "DERIVE"

  def collect(self):
    return psutil.disk_io_counters().read_bytes



class AvogadroDiskWriteBytesAgent(AvogadroAgent):
  name = "DiskWriteBytes"
  datasourceType = "DERIVE"

  def collect(self):
    return psutil.disk_io_counters().write_bytes



class AvogadroDiskReadTimeAgent(AvogadroAgent):
  name = "DiskReadTime"
  datasourceType = "DERIVE"

  def collect(self):
    return psutil.disk_io_counters().read_time



class AvogadroDiskWriteTimeAgent(AvogadroAgent):
  name = "DiskWriteTime"
  datasourceType = "DERIVE"

  def collect(self):
    return psutil.disk_io_counters().write_time
