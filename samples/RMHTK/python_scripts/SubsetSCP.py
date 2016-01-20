# author         - cz277@cam.ac.uk
# script version - 1.0
# python version - 2.7

import os
import sys
import random
import logging
import argparse

parser = argparse.ArgumentParser(description = 'select a subset of the SCP file')
egroup = parser.add_mutually_exclusive_group()
egroup.add_argument('-v', '--verbose', help = 'increase output verbosity', action = 'store_true', default = False)
egroup.add_argument('-q', '--quiet', help = 'enable silent mode', action = 'store_true', default = False)
parser.add_argument('-s', '--speaker', help = 'speaker based selection', action = 'store_true', default = False)
parser.add_argument('-m', '--speakermask', help = 'set the speaker mask', type = str, action = 'store', default = '%%%%%%*')
parser.add_argument('-r', '--rand', help = 'does random selection', action = 'store_true', default = False)
parser.add_argument('-p', '--percent', help = 'selection this percent of data', type = float, action = 'store', default = 0.1)
parser.add_argument('-o', '--order', help = 'sort output file by order', action = 'store_true', default = False)
parser.add_argument('inSCP', help = 'input SCP file path', type = argparse.FileType('r'), action = 'store')
parser.add_argument('outSCP', help = 'output SCP file path', type = argparse.FileType('w'), action = 'store')

args = parser.parse_args()
logging.basicConfig(level = logging.WARNING)
logger = logging.getLogger(__name__)
if args.quiet:
        logger.setLevel(logging.ERROR)
elif args.verbose:
        logger.setLevel(logging.INFO)

def PrintMessage(msgtype, strmsg):
        if msgtype == logging.ERROR:
                logger.error(strmsg)
                os._exit(1)
        elif msgtype == logging.WARNING:
                logger.warning(strmsg)
        else:
                logger.info(strmsg)

def MaskStr(inMask, inStr):
        outStr = ''
        if inMask.startswith('*'):      # reverse
                if inMask.endswith('*'):
                        return inStr
                for idx in range(0, len(inMask)):
                        maskIdx = len(inMask) - idx - 1
                        strIdx = len(inStr) - idx - 1
                        if inMask[maskIdx] == '%':
                                outStr += inStr[strIdx]
                        elif inMask[maskIdx] == '?':
                                outStr += ''
                        elif inMask[maskIdx] == '*':
                                break
                        else:
                                if inMask[maskIdx] != inStr[strIdx]:
                                        PrintMessage(logging.ERROR, inStr + ' does not match mask ' + inMask)
                                        os._exit(1)
        else:   # original
                for idx in range(0, len(inMask)):
                        maskIdx = idx
                        strIdx = idx
                        if inMask[maskIdx] == '%':
                                outStr += inStr[strIdx]
                        elif inMask[maskIdx] == '?':
                                outStr += ''
                        elif inMask[maskIdx] == '*':
                                break
                        else:
                                if inMask[maskIdx] != inStr[strIdx]:
                                        PrintMessage(logging.ERROR, inStr + ' does not match mask ' + inMask)
                                        os._exit(1)
        return outStr


speakerDict = {}
speakerList = []

# check the range of parameters
if args.percent <= 0.0 or args.percent > 1.0:
	PrintMessage(logging.ERROR, 'percent should set to between 0.0 to 1.0')

# load input scp
inlines = args.inSCP.readlines()

for eachline in inlines:
	eachline = eachline.replace(os.linesep, '')
	uttername = eachline.split('=')[0].split(os.sep)[-1]
	curspeaker = MaskStr(args.speakermask, uttername)
	if speakerDict.has_key(curspeaker):
		speakerDict[curspeaker].append(eachline)
	else:
		speakerList.append(curspeaker)
		speakerDict[curspeaker] = [eachline]

outlines = []
if args.speaker:
	indexList = []
	for idx in range(0, len(speakerList)):
		indexList.append(idx)
	if args.rand:
		random.shuffle(indexList)
	#nselect = int(len(speakerList) * args.percent + 1)
	nselect = int(len(speakerList) * args.percent)
	while len(indexList) > nselect:
		indexList.pop()
	indexList.sort()
	for idx in range(0, nselect):
		outlines.extend(speakerDict[speakerList[indexList[idx]]])
else:
	for eachspeaker in speakerList:
		indexList = []
		for idx in range(0, len(speakerDict[eachspeaker])):
			indexList.append(idx)
		if args.rand:
			random.shuffle(indexList)
		#nselect = int(len(speakerDict[eachspeaker]) * args.percent + 1)
		nselect = int(len(speakerDict[eachspeaker]) * args.percent)
		while len(indexList) > nselect:
			indexList.pop()
		indexList.sort()
		for idx in range(0, nselect):
			outlines.append(speakerDict[eachspeaker][indexList[idx]])

for eachline in outlines:
	args.outSCP.write(eachline + os.linesep)


