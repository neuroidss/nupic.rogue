#!/usr/bin/python
import os
import time, datetime

from AppKit import NSApplication, NSApp
from Foundation import NSObject
from PyObjCTools import AppHelper
from Cocoa import NSKeyDownMask, NSKeyUpMask, NSKeyUp, NSKeyDown, NSEvent



class AppDelegate(NSObject):
  def applicationDidFinishLaunching_(self, notification):
    mask = (NSKeyDownMask
            | NSKeyUpMask)
    NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, handler)


_BUCKET_SIZE_SECONDS = 30.0



class Helper(object):
  num = 0
  lastDownTS = None
  lastUpTS = None
  hold = []
  upDown = []
  downDown = []

  bucketStartTime = None

  # If the user is not typing for this long, we assume
  # she is doing something else entirely, not that she's a super slow typist
  inactivityThreshold = 5.0

  # How long the user is active within the bucket. We subtract inactivity
  # periods from this total.
  # At the end of the bucket, the number of keys pressed will be divided by this
  # totalActiveTime.
  totalActiveTime = _BUCKET_SIZE_SECONDS


  @classmethod
  def reset(cls):
    cls.num = 0
    cls.hold = []
    cls.upDown = []
    cls.downDown = []
    cls.bucketStartTime = time.time()
    cls.totalActiveTime = _BUCKET_SIZE_SECONDS


  @classmethod
  def keyDown(cls):
    ts = time.time()

    if cls.lastDownTS is not None:
      timeDiff = ts - cls.lastDownTS
      cls.downDown.append(timeDiff)

    if cls.lastUpTS is not None:
      timeDiff = ts - cls.lastUpTS
      cls.upDown.append(timeDiff)

    cls.lastDownTS = ts
    cls.num += 1


  @classmethod
  def keyUp(cls):
    ts = time.time()

    if cls.lastDownTS is not None:
      timeDiff = ts - cls.lastDownTS
      cls.hold.append(timeDiff)

    cls.lastUpTS = ts


  @classmethod
  def keySpeed(cls):
    ts = time.time()

    # How long has it been since the last keyUp
    if cls.lastDownTS is not None:
      timeDiff = ts - cls.lastDownTS
    else: # lastDownTS is none
      # There's been no activity since the beginning of the bucket
      timeDiff = ts - cls.bucketStartTime

    # if the time is significant, subtract it from the totalActiveTime.
    # This effectively cuts out the time the user is away from the keyboard,
    # as long as it's under our (subjectively set) threshold
    if timeDiff > cls.inactivityThreshold:
      cls.totalActiveTime -= timeDiff



def handler(event):
  """The handler for cocoa events. Only keyboard events are recorded.

  :param event: A cocoa event sent by the AppDelegate
  """
  try:
    if event.type() is NSKeyDown:
      Helper.keyDown()
    elif event.type() is NSKeyUp:
      Helper.keyUp()
  except:
    pass



def record():
  path = os.path.dirname(os.path.realpath(__file__))

  # DownDown is the time between KeyA being pressed down to KeyB being pressed
  # down, including the intermediate time
  if len(Helper.downDown) != 0:
    averageDownDown = sum(Helper.downDown)/len(Helper.downDown)
  else:
    averageDownDown = 0

  # UpDown is the time from KeyA being released to KeyB being pressed down
  # Is the inverse of KeyCount
  if len(Helper.upDown) != 0:
    averageUpDown = sum(Helper.upDown)/len(Helper.upDown)
  else:
    averageUpDown = 0

  # Hold is the time a key is held down
  if len(Helper.hold) != 0:
    averageHold = sum(Helper.hold)/len(Helper.hold)
  else:
    averageHold = 0

  # Subtract inactive time at the end of the bucket by calling this once more
  Helper.keySpeed()

  # Divide the number of keys pressed by the time the user was active
  # Will be 0 if no keys are pressed, will be max if the most keys are pressed
  # in the shortest time
  # example:
  # if a user presses 60 keys in the first 10 seconds of the 30 second bucket,
  # the inactive 20 seconds will be subtracted from totalActiveTime by the
  # manual Helper.keySpeed() call. The final keySpeed will be 60/10 = 6
  # (rather than 60/30 = 2 as it would be without subtracting the inactive time)
  keySpeed = Helper.num / Helper.totalActiveTime

  # Save the files
  with open(path + "/keys.temp", 'w') as tempFile:
    tempFile.write("%s,%s,%s,%s,%s\n" % (Helper.num, averageDownDown,
                                      averageUpDown, averageHold, keySpeed))

  timestamp = time.mktime(datetime.datetime.utcnow().timetuple())
  with open(path + "/keycounts.csv", 'ab') as csvFile:
    csvFile.write("%d,%s,%s,%s,%s,%s\n" % (timestamp, Helper.num, averageDownDown,
                                        averageUpDown, averageHold, keySpeed))

  Helper.reset()
  AppHelper.callLater(_BUCKET_SIZE_SECONDS, record)



def main():
  app = NSApplication.sharedApplication()
  delegate = AppDelegate.alloc().init()
  NSApp().setDelegate_(delegate)
  AppHelper.callLater(_BUCKET_SIZE_SECONDS, record)
  AppHelper.runEventLoop()



if __name__ == '__main__':
  main()
