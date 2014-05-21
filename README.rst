Numenta Rogue
=============

Metrics collection agent for the Numenta Rogue showcase application and
consists of two primary components: A long-running metric collection agent,
which periodically polls various metrics and records the results to a local
database, and a separate process for forwarding metrics to a Grok server for
analysis.

Each metric collection agent is executed in a continuously-running gevent
Greenlet, with a pre-determined wait interval between samples.  Each sample
is cached locally in a dedicated RRDTool database.  Individual metrics are
implemented as subclasses of `avogadro.agent.AvogadroAgent()`, and must
implement a `collect()` method, which returns the metric value.

The `rogue-forward` process fetches metric data from the individual RRDTool
databases using the `rrdtool fetch` command, and sends the metric data using
the Grok Custom Metric API.  Each time the `rogue-forward` process is
executed, the most recent timestamp is cached locally, and is used as a
starting point for subsequent requests.

Usage
-----

Install RRDTool::

    brew install rrdtool

Install in development mode::

    python setup.py develop --install-dir=... --script-dir=...

Start key-counter with nohup::

    nohup rogue-keycounter > rogue-keys.stdout 2> rogue-keys.stderr < /dev/null &

Make sure that iTerm/Terminal is allowed in System Prefrences > Security & Privacy > Privacy > Accessibility!

Start metric collection agent with nohup::

    nohup rogue-agent --prefix=var/db/ --interval=300 --hearbeat=600 > rogue-agent.stdout 2> rogue-agent.stderr < /dev/null &

Forward pending metric data to Grok once::

    rogue-forward --server=https://localhost --prefix=var/db

Sample crontab entry::

    * * * * * PATH=$PATH:/usr/local/bin PYTHONPATH=... .../rogue-forward --server=... --prefix=... > .../rogue-forward.stdout 2> .../rogue-forward.stderr < /dev/null

Export to local CSV
~~~~~~~~~~~~~~~~~~~

All metric data is written to a local round-robin database, which only retains
the two most recent weeks of data at any given moment in time.  Should you want
the data exported to CSV, use the `rogue-export` utility::

    rogue-export --prefix=var/db

The exporter keeps track of the position of a given metric, so you can run
`rogue-export` as frequently as you like, and the `.csv` files in `var/db/`,
will be updated accordingly.  i.e. you can periodically sync the round-robin
database to a csv file in `var/db/`


Metrics
-------

CPUPercent
~~~~~~~~~~

Total CPU utilization as a percentage, as reported by ``psutil.cpu_percent()``.

MemoryPercent
~~~~~~~~~~~~~

The percentage memory usage calculated as `(total - available) / total * 100`,
as reported by ``psutil.virtual_memory().percent``

DiskReadBytes
~~~~~~~~~~~~~

Number of bytes read (total), as reported by
``psutil.disk_io_counters.read_bytes``

DiskWriteBytes
~~~~~~~~~~~~~

Number of bytes written (total), as reported by
``psutil.disk_io_counters.write_bytes``

DiskReadTime
~~~~~~~~~~~~

Time spent reading from disk (in milliseconds), as reported by
``psutil.disk_io_counters.read_time``

DiskWriteTime
~~~~~~~~~~~~~

Time spent writing to disk (in milliseconds), as reported by
``psutil.disk_io_counters.write_time``

NetworkBytesSent
~~~~~~~~~~~~~~~~

Number of bytes sent, as reported by ``psutil.net_io_counters.bytes_sent()``

NetworkBytesReceived
~~~~~~~~~~~~~~~~~~~~

Number of bytes received, as reported by
``psutil.net_io_counters.bytes_recv()``

Copyright 2013-2014 Numenta Inc.
--------------------------------

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

