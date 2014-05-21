#!/usr/bin/env python
# This converts data from the individual 4 Key files outputted by avogadro

import datetime, time, sys, csv, os.path
import contextlib

try:
  inputPath = sys.argv[1]
  outputPath = sys.argv[2]
except IndexError:
  print ("Usage: python %s path/to/input/ path/to/output/" % sys.argv[0])
  exit()

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

countInputFilePath = inputPath + "KeyCount.csv"
ddInputFilePath = inputPath + "KeyDownDown.csv"
udInputFilePath = inputPath + "KeyUpDown.csv"
holdInputFilePath = inputPath + "KeyHold.csv"

countOutputFilePath = outputPath + "Count.csv"
ddOutputFilePath = outputPath + "DownDown.csv"
udOutputFilePath = outputPath + "UpDown.csv"
holdOutputFilePath = outputPath + "Hold.csv"

totalData = []

# Open all the input files
with open(countInputFilePath, "rb") as countInputFile, \
     open(ddInputFilePath, "rb") as ddInputFile, \
     open(udInputFilePath, "rb") as udInputFile,\
     open(holdInputFilePath, "rb") as holdInputFile:

  countReader = csv.reader(countInputFile)
  ddReader = csv.reader(ddInputFile)
  udReader = csv.reader(udInputFile)
  holdReader = csv.reader(holdInputFile)

  # Get total number of rows
  numRows = sum(1 for row in countReader)

  # Reset the file pointer (there should be a better way to do this...)
  countInputFile.close()
  countInputFile = open(countInputFilePath, "rb")
  countReader = csv.reader(countInputFile)

  for _ in xrange(numRows):
    countRow = countReader.next()
    ddRow = ddReader.next()
    udRow = udReader.next()
    holdRow = holdReader.next()

    # Timestamp, count, dd, ud, hold
    totalData.append([countRow[2], countRow[1], ddRow[1], udRow[1], holdRow[1]])

# Now we have all the data in one place
# Write it to separate files in the formatted directory for processing by
# process_keys.py
lastTS = None

with open(countOutputFilePath, "wb") as countFile, \
        open(ddOutputFilePath, "wb") as ddFile, \
        open(udOutputFilePath, "wb") as udFile,\
        open(holdOutputFilePath, "wb") as holdFile:

    countWriter = csv.writer(countFile)
    ddWriter = csv.writer(ddFile)
    udWriter = csv.writer(udFile)
    holdWriter = csv.writer(holdFile)

    headerRow = ["name", "value", "dttm"]
    countWriter.writerow(headerRow)
    ddWriter.writerow(headerRow)
    udWriter.writerow(headerRow)
    holdWriter.writerow(headerRow)

    for row in totalData:
      ts = int(float(row[0])) / 300 * 300

      readableTS = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

      count = row[1]
      dd = row[2]
      ud = row[3]
      hold = row[4]

      #Bugs caused by lid closing
      if float(hold) > 1.0:
        continue
      elif float(dd) > 300 or float(ud) > 300:
        continue

      if float(dd) > float(ud):
        countWriter.writerow(["KeyCount", count, readableTS])
        ddWriter.writerow(["KeyDownDown", "nan", readableTS])
        udWriter.writerow(["KeyUpDown", "nan", readableTS])
        holdWriter.writerow(["KeyHold", hold, readableTS])
        lastTS = ts
        continue

      if lastTS is None:
        countWriter.writerow(["KeyCount", count, readableTS])
        ddWriter.writerow(["KeyDownDown", dd, readableTS])
        udWriter.writerow(["KeyUpDown", ud, readableTS])
        holdWriter.writerow(["KeyHold", hold, readableTS])
        lastTS = ts
        continue

      if (ts - lastTS) > 300:
        while (ts - lastTS > 300):
          lastTS += + 300
          readableLastTS = datetime.datetime.fromtimestamp(lastTS).strftime('%Y-%m-%d %H:%M:%S')
          countWriter.writerow(["KeyCount", "nan", readableLastTS])
          ddWriter.writerow(["KeyDownDown", "nan", readableLastTS])
          udWriter.writerow(["KeyUpDown", "nan", readableLastTS])
          holdWriter.writerow(["KeyHold", "nan", readableLastTS])

      if (ts - lastTS) == 300:
        countWriter.writerow(["KeyCount", count, readableTS])
        ddWriter.writerow(["KeyDownDown", dd, readableTS])
        udWriter.writerow(["KeyUpDown", ud, readableTS])
        holdWriter.writerow(["KeyHold", hold, readableTS])

      lastTS = ts
