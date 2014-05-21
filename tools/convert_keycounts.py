import datetime, time, sys, csv, os.path
import contextlib

inputPath = os.path.dirname(os.path.abspath(__file__)) + "/"

try:
  inputFilePath = sys.argv[1]
  outputPath = sys.argv[2]
except IndexError:
  print ("Usage: python %s infilename.csv path/to/output/folder/" % sys.argv[0])
  exit()

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

print outputPath

countOutputFilePath = outputPath + "Count.csv"
ddOutputFilePath = outputPath + "DownDown.csv"
udOutputFilePath = outputPath + "UpDown.csv"
holdOutputFilePath = outputPath + "Hold.csv"

lastTS = None


with open(inputFilePath, "rb") as inputFile:
  csvreader = csv.reader(inputFile)

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

    for row in csvreader:
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
