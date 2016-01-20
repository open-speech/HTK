# author         - cz277@cam.ac.uk
# script version - 1.0
# python version - 2.7

import os
import sys
import logging
import argparse

parser = argparse.ArgumentParser(description = 'convert extended SCP file')
egroup = parser.add_mutually_exclusive_group()
egroup.add_argument('-v', '--verbose', help = 'increase output verbosity', action = 'store_true', default = False)
egroup.add_argument('-q', '--quiet', help = 'enable silent mode', action = 'store_true', default = False)
parser.add_argument('-s', '--side', help = 'generate side based SCP', action = 'store_true', default = False)
parser.add_argument('-sm', '--sidemask', help = 'set the mask for data side', type = str, action = 'store', default = '%%%%%%%%%%%%????????????????%%%_*')
parser.add_argument('-r', '--reorder', help = 'sort the output SCP', action = 'store_true', default = False)
parser.add_argument('-c', '--hcopy', help = 'generate HCopy style SCP', action = 'store_true', default = False)
parser.add_argument('-dm', '--dirmask', help = 'set the mask to convert input to output directory', type = str, action = 'store', default = '')
parser.add_argument('-u', '--update', help = 'update the segment boundary fields', action = 'store_true', default = False)
parser.add_argument('-o', '--outbase', help = 'set output base directory', type = str, action = 'store', default = '')
parser.add_argument('-n', '--notext', help = 'generate non-extended SCP file', action = 'store_true', default = False)
parser.add_argument('-x', '--dataext', help = 'set the data extension', type = str, action = 'store', default = '')
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


def CheckDir(inDir):
	if inDir == '':
		return ''
	elif not inDir.endswith(os.sep):
		inDir += os.sep
	return inDir


def MaskStr(inMask, inStr):
	outStr = ''
	if inMask.startswith('*'):	# reverse
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
	else:	# original
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
	

def UpdateRangeField(line):
	item = line.split('[')[-1]
	if item.count(']') == 0:
		return ''
	else:
		parts = item.split(']')[0].split(',')
		start = int(parts[0])
		end = int(parts[1])
		end = end - start
		start = 0
		return "[%06d,%06d]" % (start, end)

def UpdateDataExt(line, dataext, extscp):
	logid = ''
	if line.count('='):
		items = line.split('=')
		logid = os.path.splitext(items[0])[0] + '.' + dataext + '='
		line = items[1]
	items = line.split(' ')
	for idx in range(0, len(items)):
		parts = items[idx].split('[')
		items[idx] = os.path.splitext(parts[0])[0] + '.' + dataext
		if extscp:
			if len(parts) > 1:
				items[idx] += '[' + parts[1]
	if extscp:
		return logid + ' '.join(items)
	else:
		return ' '.join(items)


# parse the options
if args.hcopy:
	if args.side and args.update:
		PrintMessage(logging.WARNING, 'Target side names will be changed in HCopy style scp')
	elif (not args.side) and (not args.update):
		args.update = True
		PrintMessage(logging.WARNING, 'For segment level HCopy, target file names are the segment names')
if args.dataext.startswith('.'):
	args.dataext = args.dataext[1: ]
		

# load input scp
inlines = args.inSCP.readlines()


# arrange the input lines
outList = []
outDict = {}
for eachline in inlines:
	eachline = eachline.replace(os.linesep, '')
	uttId = eachline.split('=')[0].split('[')[0].split(os.sep)[-1]
	if args.side:
		uttId = MaskStr(args.sidemask, uttId)
	if not outDict.has_key(uttId):
		outList.append(uttId)
		if args.side:
			outDict[uttId] = eachline.split('[')[0]
		else:
			outDict[uttId] = eachline


# formatting the output lines
outlines = []
for eachId in outList:
	curline = ''
	src = outDict[eachId]
	srcDir = CheckDir(os.path.dirname(src.split('=')[-1]))
	srcSub = CheckDir(MaskStr(args.dirmask, srcDir))
	if args.outbase == '':
		tgt = srcDir + srcSub
	else:
		tgt = CheckDir(args.outbase) + srcSub
	if args.update:
		tgt += eachId
	else:
		tgt += src.split(os.sep)[-1].split('[')[0]
	if args.hcopy:
		#outlines.append(src + ' ' + tgt + os.linesep)
		curline = src + ' ' + tgt
	else:
		if args.update:
			tgt += UpdateRangeField(src)
		elif (not args.side) and src.count('['):
			tgt += '[' + src.split('[')[-1]
		if args.notext:
			curline = tgt
		else:
			curline = eachId + '=' + tgt
	if args.dataext != '':
		curline = UpdateDataExt(curline, args.dataext, not args.notext)
	outlines.append(curline + os.linesep)

if args.reorder:
	outlines.sort()
	
for eachline in outlines:
	args.outSCP.write(eachline)


