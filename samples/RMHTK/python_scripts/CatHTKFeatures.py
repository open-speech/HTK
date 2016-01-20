# author         - cz277@cam.ac.uk
# script version - 1.0
# python version - 2.7

import os
import sys
import struct
import logging
import argparse


parser = argparse.ArgumentParser(description = 'concatenate HTK binary features')
egroup = parser.add_mutually_exclusive_group()
egroup.add_argument('-v', '--verbose', help = 'increase output verbosity', action = 'store_true', default = False)
egroup.add_argument('-q', '--quiet', help = 'enable silent mode', action = 'store_true', default = False)
parser.add_argument('-l', '--little', help = 'use little-endian rather than big-endian', action = 'store_true', default = False)
parser.add_argument('-p', '--parmkind', help = 'the output feature parmKind', type = str, action = 'store', default = 'USER')
parser.add_argument('inLFEA', help = 'input left hand HTK feature file', type = argparse.FileType('rb'), action = 'store')
parser.add_argument('inRFEA', help = 'input right hand HTK feature file', type = argparse.FileType('rb'), action = 'store')
parser.add_argument('outFEA', help = 'output combined HTK feature file', type = argparse.FileType('wb'), action = 'store')

typelist = ['WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC', 'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP']
typedict = {}
for idx in range(0, len(typelist)):
	typedict[typelist[idx]] = idx 
qualdict = {'_E': 0o000100, '_N': 0o000200, '_D': 0o000400, '_A': 0o001000, '_C': 0o002000, '_Z': 0o004000, '_K': 0o010000, '_0': 0o020000}

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

def ReadHTKFeature(feafile, order):
	# load the feature file head
	nSamples = feafile.read(4)
	sampPeriod = feafile.read(4)
	sampSize = feafile.read(2)
	parmKind = feafile.read(2)
	# unpack the numbers
	nSamples = struct.unpack(order + 'I', nSamples)[0]
	sampPeriod = struct.unpack(order + 'I', sampPeriod)[0]
	sampSize = struct.unpack(order + 'H', sampSize)[0]
	sampleDim = sampSize / 4
	parmKind = struct.unpack(order + 'H', parmKind)[0]
	if parmKind & qualdict['_C'] != 0 or parmKind & qualdict['_K'] != 0:
		PrintMessage(logging.ERROR, '_C and _K not supported, use HCopy to convert first')
	header = [nSamples, sampPeriod, sampSize, parmKind]
	# read each value
	values = []
	for ridx in range(0, nSamples):
		values.append([])
		for cidx in range(0, sampleDim):
			curval = feafile.read(4)
			curval = struct.unpack(order + 'f', curval)[0]
			values[-1].append(curval)
	return [header, values]

def WriteHTKFeature(feafile, order, feature):
	# write the header
	header = feature[0]
	feafile.write(struct.pack(order + 'I', header[0]))
	feafile.write(struct.pack(order + 'I', header[1]))
	feafile.write(struct.pack(order + 'H', header[2]))
	feafile.write(struct.pack(order + 'H', header[3]))
	if header[3] & qualdict['_C'] != 0 or header[3] & qualdict['_K'] != 0:
		PrintMessage(logging.ERROR, '_C and _K not supported, reset with --parmkind')
	# write the values
	values = feature[1]
	for ridx in range(0, len(values)):
		for cidx in range(0, len(values[ridx])):
			feafile.write(struct.pack(order + 'f', values[ridx][cidx]))

def ShowHTKFeaHeader(feature):
	header = feature[0]
	# nSamples
	PrintMessage(logging.INFO, 'nSamples: ' + str(header[0]))
	# sampPeriod
	PrintMessage(logging.INFO, "sampPeriod: %e" % float(header[1]))
	# sampSize
	PrintMessage(logging.INFO, 'sampleDim ' + str(header[2] / 4))
	# parmKind
	parmkind = typelist[header[3] - int(header[3] / 64) * 64]
	for (key, value) in qualdict.items():
		if header[3] & value != 0:
			parmkind += key
	PrintMessage(logging.INFO, 'parmKind: ' + parmkind)

def GetParmKindInt(parmkind):
	intkind = 0
	items = parmkind.split('_')
	intkind = intkind | typedict[items[0]]
	for idx in range(1, len(items)):
		curqual = '_' + items[idx]
		intkind = intkind | qualdict[curqual]
	return intkind


# big endian by default
order = '<'
if args.little:
	order = '>'

# read left hand HTK feature
lhFea = ReadHTKFeature(args.inLFEA, order)
PrintMessage(logging.INFO, 'Left hand input feature:')
ShowHTKFeaHeader(lhFea)
rhFea = ReadHTKFeature(args.inRFEA, order)
PrintMessage(logging.INFO, 'Right hand input feature:')
ShowHTKFeaHeader(rhFea)

if lhFea[0][0] != rhFea[0][0]:
	PrintMessage(logging.ERROR, 'left and right features have different sample numbers')
if lhFea[0][1] != rhFea[0][1]:
	PrintMessage(logging.ERROR, 'left and right features have different sample rates')

# generate the out feature header
header = [lhFea[0][0], lhFea[0][1], lhFea[0][2] + lhFea[0][2], GetParmKindInt(args.parmkind.upper())]
values = []
for idx in range(0, len(lhFea[1])):
	values.append([])
	values[-1].extend(lhFea[1][idx])
	values[-1].extend(rhFea[1][idx])
catFea = [header, values]

# write-out the combined feature
PrintMessage(logging.INFO, 'Concatenated output feature vector:')
ShowHTKFeaHeader(catFea)
WriteHTKFeature(args.outFEA, order, catFea)


