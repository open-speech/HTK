# author         - cz277@cam.ac.uk
# script version - 1.1
# python verison - 2.7

import os
import sys
import types
import logging
import argparse

cmdparser = argparse.ArgumentParser(description = 'generate initial HTK V3.5 DNN model')
egroup = cmdparser.add_mutually_exclusive_group()
egroup.add_argument('-v', '--verbose', help = 'increase output verbosity', action = 'store_true', default = False)
egroup.add_argument('-q', '--quiet', help = 'enable silent mode', action = 'store_true', default = False)
cmdparser.add_argument('-s', '--structure', help = 'the initial DNN structure', type = str, action = 'store', default = '')
cmdparser.add_argument('inHTE', help = 'input HTE config file path', type = argparse.FileType('r'), action = 'store')
cmdparser.add_argument('outMMF', help = 'output initial DNN file path', type = argparse.FileType('w'), action = 'store')
args = cmdparser.parse_args()

actdict = {'AFFINE': [1.0, 0.0], 'HERMITE': [0.0] * 10, 'LINEAR': [], 'RELU': [], 'PRELU': [1.0], 
           'PARMRELU': [1.0, 0.25], 'SIGMOID': [], 'LHUCSIGMOID': [1.0], 'PSIGMOID': [1.0], 
           'PARMSIGMOID': [1.0, 1.0, 0.0], 'SOFTRELU': [], 'SOFTMAX': [], 'TANH': []}
cfglist = ['DNNSTRUCTURE', 'FEATURETYPE', 'FEATUREDIM', 'CONTEXTSHIFT', 'HIDDENACTIVATION', 'OUTPUTACTIVATION']
cfgdict = {'DNNSTRUCTURE': '720X1000X3000', 'FEATURETYPE': '<FBANK_D_Z>', 'FEATUREDIM': '80',
           'CONTEXTSHIFT': '-4,-3,-2,-1,0,1,2,3,4', 'HIDDENACTIVATION': 'SIGMOID', 'OUTPUTACTIVATION': 'SOFTMAX'}

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('GenInitDNN')
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

def PackOneWord(left, word, right):
	if not word.startswith(left):
		word = left + word
	if not word.endswith(right):
		word += right
	return word

def ParseHTE(lines):
	for eachline in lines:
		curline = eachline.replace(os.linesep, '').replace('\t', ' ').split('#')[0]
		while curline.count('  '):
			curline = curline.replace('  ', ' ')
		if curline.startswith(' '):
			curline = curline[1: ]
		elif curline.endswith(' '):
			curline = curline[: -1]
		if curline == '':
			continue
		if curline.startswith('set '):
			curline = curline[4: ]
		curline = curline.replace(' = ', '=')
		fields = curline.split('=')
		if len(fields) != 2:
			PrintMessage(logging.WARNING, 'Ignore illegal config ' + curline)
			continue
		fields[0] = fields[0].upper()
		if not cfgdict.has_key(fields[0]):
			cfglist.append(fields[0])
		cfgdict[fields[0]] = fields[1]
	cfglist.sort()

def GenNVecBundle(nlen, value):
	return [value] * nlen

def GenNMatBundle(nrows, ncols, value):
	return [[value] * ncols] * nrows

def Vector2Str(vector):
	retstr = ''
	for eachval in vector:
		if type(eachval) is types.FloatType:
			retstr += " %e" % eachval
		elif type(eachval) is types.IntType:
			retstr += " %d" % eachval
		else:
			retstr += " " + str(eachval)
	return retstr

def Matrix2Strs(matrix):
	retstrs = []
	for eachrow in matrix:
		curstr = ''
		for eachval in eachrow:
			curstr += " %e" % eachval
		retstrs.append(curstr)
	return retstrs

def GetLayerID(lidx, lnum):
	if lidx == lnum - 1:
		return 'out'
	elif lidx < 0:
		return 'in'
	else:
		return str(lidx + 2)

# 1. load input HTE file
ParseHTE(args.inHTE.readlines())
if args.structure != '':
	cfgdict['DNNSTRUCTURE'] = args.structure
cfgdict['FEATURETYPE'] = PackOneWord('<', cfgdict['FEATURETYPE'], '>')
cfgdict['CONTEXTSHIFT'] = cfgdict['CONTEXTSHIFT'].split(',')
cfgdict['HIDDENACTIVATION'] = cfgdict['HIDDENACTIVATION'].upper()
cfgdict['OUTPUTACTIVATION'] = cfgdict['OUTPUTACTIVATION'].upper()
for eachcfg in cfglist:
	value = cfgdict[eachcfg]
	PrintMessage(logging.INFO, eachcfg + '\t= ' + str(value))

# 2. parse the ANN structure
layerdims = cfgdict['DNNSTRUCTURE'].split('X')
layerinfo = []
for index in range(0, len(layerdims) - 2):
	layerinfo.append([])
	layerinfo[-1].append(int(layerdims[index + 1]))
	layerinfo[-1].append(int(layerdims[index]))
	layerinfo[-1].append(cfgdict['HIDDENACTIVATION'])
layerinfo.append([int(layerdims[-1]), int(layerdims[-2]), cfgdict['OUTPUTACTIVATION']])
if layerinfo[0][1] != int(cfgdict['FEATUREDIM']) * len(cfgdict['CONTEXTSHIFT']):
	layerinfo[0][1] = int(cfgdict['FEATUREDIM']) * len(cfgdict['CONTEXTSHIFT'])
	PrintMessage(logging.WARNING, 'Unmatched input dimension, reset to ' + str(layerinfo[0][1]))

# 3. output the model
outlines = []

# generate ~o
outlines.append('~o')
outlines.append('<STREAMINFO> 1 ' + str(int(cfgdict['FEATUREDIM'])))
outlines.append('<VECSIZE> ' + str(int(cfgdict['FEATUREDIM'])) + '<NULLD>' + cfgdict['FEATURETYPE'] + '<DIAGC>')

# generate ~V
for lidx in range(0, len(layerinfo)):
	nodenum = layerinfo[lidx][0]
	function = layerinfo[lidx][2]
	outlines.append('~V ' + PackOneWord('"', "layer%s_bias" % GetLayerID(lidx, len(layerinfo)), '"'))
	outlines.append('<VECTOR> ' + str(nodenum))
	outlines.append(Vector2Str(GenNVecBundle(nodenum, 0.0)))
	for vidx in range(0, len(actdict[function])):
		outlines.append('~V ' + PackOneWord('"', "layer%s_actparam%d" % (GetLayerID(lidx, len(layerinfo)), vidx + 1), '"'))
		outlines.append('<VECTOR> ' + str(nodenum))
		outlines.append(Vector2Str(GenNVecBundle(nodenum, actdict[function][vidx])))

# generate ~M
for lidx in range(0, len(layerinfo)):
	nodenum = layerinfo[lidx][0]
	inputdim = layerinfo[lidx][1]
	outlines.append('~M ' + PackOneWord('"', "layer%s_weight" % GetLayerID(lidx, len(layerinfo)), '"'))
	outlines.append('<MATRIX> ' + str(nodenum) + ' ' + str(inputdim))
	outlines.extend(Matrix2Strs(GenNMatBundle(nodenum, inputdim, 0.0)))

# generate ~F
for lidx in range(0, len(layerinfo)):
	nodenum = layerinfo[lidx][0]
	inputdim = layerinfo[lidx][1]
	outlines.append('~F ' + PackOneWord('"', "layer%s_feamix" % GetLayerID(lidx - 1, len(layerinfo)), '"'))
	outlines.append('<NUMFEATURES> 1 ' + str(inputdim))
	if lidx == 0:
		outlines.append('<FEATURE> 1 ' + str(cfgdict['FEATUREDIM']))
		outlines.append('<SOURCE>')
		outlines.append(cfgdict['FEATURETYPE'])
		outlines.append('<CONTEXTSHIFT> ' + str(len(cfgdict['CONTEXTSHIFT'])))
		outlines.append(Vector2Str(cfgdict['CONTEXTSHIFT']))
	else:
		outlines.append('<FEATURE> 1 ' + str(inputdim))
		outlines.append('<SOURCE>')
		outlines.append('~L ' + PackOneWord('"', "layer%s" % GetLayerID(lidx - 1, len(layerinfo)), '"'))
		outlines.append('<CONTEXTSHIFT> 1')
		outlines.append(' 0')

# generate ~L
for lidx in range(0, len(layerinfo)):
	function = layerinfo[lidx][2]
	outlines.append('~L "layer' + GetLayerID(lidx, len(layerinfo)) + '"')
	outlines.append('<BEGINLAYER>')
	outlines.append('<LAYERKIND> ' + PackOneWord('"', "PERCEPTRON", '"'))
	outlines.append('<INPUTFEATURE>')
	outlines.append('~F ' + PackOneWord('"', "layer%s_feamix" % GetLayerID(lidx - 1, len(layerinfo)), '"'))
	outlines.append('<WEIGHT>')
	outlines.append('~M ' + PackOneWord('"', "layer%s_weight" % GetLayerID(lidx, len(layerinfo)), '"'))
	outlines.append('<BIAS>')
	outlines.append('~V ' + PackOneWord('"', "layer%s_bias" % GetLayerID(lidx, len(layerinfo)), '"'))
	outlines.append('<ACTIVATION> "' + function + '"')
	if len(actdict[function]) > 0:
		outlines.append('<NUMPARAMETERS> ' + str(len(actdict[function])))
	for vidx in range(0, len(actdict[function])):
		outlines.append('<PARAMETER> ' + str(vidx + 1))
		outlines.append('~V ' + PackOneWord('"', "layer%s_actparam%d" % (GetLayerID(lidx, len(layerinfo)), vidx + 1), '"'))
	outlines.append('<ENDLAYER>')

# generate ~N
outlines.append('~N ' + PackOneWord('"', "DNN1", '"'))
outlines.append('<BEGINANN>')
outlines.append('<NUMLAYERS> ' + str(len(layerinfo) + 1))
for lidx in range(0, len(layerinfo)):
	outlines.append('<LAYER> ' + str(lidx + 2))
	outlines.append('~L ' + PackOneWord('"', "layer%s" % GetLayerID(lidx, len(layerinfo)), '"'))
outlines.append('<ENDANN>')

# 4. write out the lines
for eachline in outlines:
	args.outMMF.write(eachline + os.linesep)


