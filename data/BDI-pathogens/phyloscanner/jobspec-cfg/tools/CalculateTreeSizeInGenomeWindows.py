#!/usr/bin/env python
from __future__ import print_function

## Author: Chris Wymant, c.wymant@imperial.ac.uk
## Acknowledgement: I wrote this while funded by ERC Advanced Grant PBDR-339251
##
## Overview:
ExplanatoryMessage = '''Splits an alignment of sequences up into (potentially
overlapping) windows, calculates a tree for each with RAxML (or IQtree), and characterises
the size of the tree by the median of patristic distances between all possible
pairs of tips. As output, tree sizes are reported by window, and also by
individual position in the genome (by taking the mean value of all trees
overlapping that position). This latter output is suitable for use for
normalising branch lengths over the genome in phyloscanner_analyse_trees.R
though its --normRefFileName option.'''

import argparse
import os
import sys
import subprocess
from Bio import AlignIO
import phyloscanner_funcs as pf

# Define a function to check files exist, as a type for the argparse.
def File(MyFile):
  if not os.path.isfile(MyFile):
    raise argparse.ArgumentTypeError(MyFile+' does not exist or is not a file.')
  return MyFile

# Set up the arguments for this script
parser = argparse.ArgumentParser(description=ExplanatoryMessage)
parser.add_argument('alignment', type=File)
parser.add_argument('ChosenSeqName', help='''The sequence whose coordinates
we'll use to define windows (it must be present in the
alignment). Assuming you are using this script to provide input for
phyloscanner_analyse_trees.R, you must use the same sequence here as you used
for phyloscanner_make_trees.py's --pairwise-align-to option: then the genome
positions will mean the same thing.''')
parser.add_argument('start', type=int, help='''The start position for
the first window, relative to your chosen sequence (i.e. '1' would mean the
position in the alignment where your chosen sequence has its first base). A
simple choice would be 1; however if your chosen sequence begins with a stretch
of sequence not expected to contain sensible phylogenetic information (such as
the LTRs for HIV virus), you should start later on in the sequence.''')
parser.add_argument('WindowWidth', type=int, help='''Window width with respect
to your chosen sequence, i.e. the number of bases each window will contain from
your chosen sequence, not including any gaps in that sequence. A sensible value
to choose would be the same as or similar to the window width you use for
phyloscanner_make_trees.py.''')
parser.add_argument('OutFileBaseName')
parser.add_argument('-E', '--end', type=int, help='''The end position for the
last window. By default this is the length of your chosen sequence; however see
the warning for the 'start' argument above.''')
parser.add_argument('-I', '--increment', type=int, help='''The increment between
the start of one window and the start of the next one. The default is one tenth
of the window width (so that, except for close to the edges of the genome,
each position is spanned by ten different windows). A smaller increment will be
used just for the final window if necessary for it to finish exactly at the
desired end point.''')
parser.add_argument('-T', '--threads', type=int, help='''Number of threads to
use. See the phyloscanner manual chapter 'Branch length normalisation' for an
explanation of an alternative way of parallelising this script that is suitable
for massive parallelisation (as opposed to just using multiple cores on a single
machine, which this option does).''')
parser.add_argument('--x-raxml', help=pf.RaxmlHelp)
parser.add_argument('--x-raxml-old', help=pf.RaxmlOldHelp)
parser.add_argument('--x-iqtree', help=pf.IQtreeHelp)
parser.add_argument('-Q', '--quiet', action='store_true', help='''Turns off the
small amount of information printed to the terminal (via stdout). We'll still
print warnings and errors (via stderr).''')

args = parser.parse_args()

# For files we'll create
FileForAlignment_basename = 'Alignment'

# Check output files don't exist
OutFileByWindow = args.OutFileBaseName + '_ByWindow.csv'
OutFileByPosition = args.OutFileBaseName + '_ByPosition.csv'
if os.path.isfile(OutFileByWindow):
  print(OutFileByWindow,
  'exists already. Move, rename or delete it. Quitting.', file=sys.stderr)
  exit(1)
if os.path.isfile(OutFileByPosition):
  print(OutFileByPosition,
  'exists already. Move, rename or delete it. Quitting.', file=sys.stderr)
  exit(1)

# Check the code
PythonPath = sys.executable
TreeSizeCode = pf.FindAndCheckCode(PythonPath,
'CalculateMedianPatristicDistance.R', IsPyCode=False)
ToPerPositionCode = pf.FindAndCheckCode(PythonPath,
'FromPerWindowStatsToPerPositionStats.py')
Use_raxml_old = args.x_raxml_old != None
Use_iqtree = args.x_iqtree != None
Use_raxml_ng = args.x_raxml != None

if Use_raxml_old + Use_iqtree + Use_raxml_ng > 1:
  print('Arguments for multiple tree inference tools detected. Quitting.')
  exit(1)
  
# Select a Test function depending on chosen tree inference program
if Use_raxml_old:
  TreeArgList = pf.TestTreeInference(args.x_raxml_old, "RAxML-standard")
elif Use_iqtree:
  TreeArgList = pf.TestTreeInference(args.x_iqtree, "IQtree")
else:
  TreeArgList = pf.TestTreeInference(args.x_raxml, "RAxML-NG")

# Set up multithreading if needed
multithread = args.threads != None
if multithread:
  if args.threads == 1:
    multithread = False
  else:
    try:
      from multiprocessing.dummy import Pool
    except ImportError:
      print('Problem importing Pool from the multiprocessing.dummy module. This',
      'is required for multithreading. Quitting.', file=sys.stderr)
      exit(1)
    if args.threads < 2:
      print('The number of threads must be positive. Quitting.',
      file=sys.stderr)
      exit(1)

    
# Extract the chosen seq
try:
  alignment = AlignIO.read(args.alignment, "fasta")
except:
  print('Problem reading in', args.alignment + '. Quitting.', file=sys.stderr)
  raise
AlignmentLength = alignment.get_alignment_length()
ChosenSeqFound = False
for seq in alignment:
  if seq.id == args.ChosenSeqName:
    ChosenSeq = seq
    ChosenSeqFound = True
    break
if not ChosenSeqFound:
  print('Did not find', args.ChosenSeqName, 'in', args.alignment +
  '. Quitting.', file=sys.stderr)
  exit(1)
GappyChosenSeq = str(ChosenSeq.seq)
UngappedChosenSeq = str(ChosenSeq.seq.ungap('-'))
ChosenSeqLen = len(UngappedChosenSeq)

# Check it contains some bases
if ChosenSeqLen == 0:
  print(args.ChosenSeqName, 'contains no bases, only gaps. Quitting.',
  file=sys.stderr)
  exit(1)

# Set the end point if not specified
EndNotSpecified = args.end == None
if EndNotSpecified:
  args.end = ChosenSeqLen

# Sanity checks on the int parameters
if not 0 < args.start < ChosenSeqLen:
  print('The start should be greater than zero and less than the length',
  ' of your chosen sequence (', ChosenSeqLen, '). Quitting.', sep='',
  file=sys.stderr)
  exit(1)
if not args.start < args.end <= ChosenSeqLen:
  print('The end should be greater than the start and at most equal to the',
  ' length of your chosen sequence (', ChosenSeqLen, '). Quitting.', sep='',
  file=sys.stderr)
  exit(1)
if not 0 < args.WindowWidth < args.end - args.start + 2:
  message = 'The window width should be greater than zero and at most equal to'\
  ' (end - start + 1), where '
  if EndNotSpecified:
    message += 'end is the length of your chosen sequence: ' + str(ChosenSeqLen)
  else:
    message += 'you specifed an end value of ' + str(args.end)
  message += '. Quitting.'
  print(message, file=sys.stderr)
  exit(1)
if args.increment == None:
  args.increment = max(args.WindowWidth / 10, 1)
elif not 0 < args.increment < args.WindowWidth + 1:
  print('The increment should be greater than 0, and at most equal to the',
  'window width (so that there is no unused space in between consecutive',
  'windows). Quitting.', file=sys.stderr)
  exit(1)

# Define all windows
WindowStarts = []
WindowEnds = []
NextStart = args.start
NextEnd = args.start + args.WindowWidth - 1
while NextEnd <= args.end:
  WindowStarts.append(NextStart)
  WindowEnds.append(NextEnd)
  NextStart += args.increment
  NextEnd = NextStart + args.WindowWidth - 1
# Add an extra window that finishes right on the end point, if the last one
# doesn't already.
if WindowEnds[-1] != args.end:
  WindowEnds.append(args.end)
  WindowStarts.append(args.end - args.WindowWidth + 1)

# Translate coordinates from being with respect to the chosen sequence to being
# with respect to the alignment
TranslatedStarts = pf.TranslateSeqCoordsToAlnCoords(GappyChosenSeq,
WindowStarts)
TranslatedEnds = pf.TranslateSeqCoordsToAlnCoords(GappyChosenSeq,
WindowEnds)
assert len(TranslatedStarts) == len(TranslatedEnds) > 0
NumWindows = len(TranslatedStarts)

# Sanity check on the translated window coords
for i in range(NumWindows):
  start = TranslatedStarts[i]
  end = TranslatedEnds[i]
  if not 0 < start < end <= AlignmentLength:
    print('Unexpected error: encountered a window with the following alignment',
    ' coordinates: ', start, '-', end, '. (The alignment is ', AlignmentLength,
    ' positions long.) Quitting.', sep='', file=sys.stderr)
    exit(1)


# Keep track of temp files to delete them at the end
TempFilesSet = set([])

def GetTreeSizeFromWindow(WindowNumber):
  '''Extracts a window from an alignement, makes a tree, finds the tree size.'''

  # Get the start and end. Zero-based indexing for the alignment.
  ChosenSeqStart = WindowStarts[WindowNumber]
  ChosenSeqEnd = WindowEnds[WindowNumber]
  start = TranslatedStarts[WindowNumber] - 1
  end = TranslatedEnds[WindowNumber] - 1

  if not args.quiet:
    print('Now processing window ' + str(ChosenSeqStart) + '-' + 
    str(ChosenSeqEnd) + '.')

  WindowSuffix = 'InWindow_'+str(ChosenSeqStart)+'_to_'+\
  str(ChosenSeqEnd)
  WindowAsStr = str(ChosenSeqStart) + '-' + str(ChosenSeqEnd)

  SeqAlignmentHere = alignment[:, start:end+1]
  FileForAlnHere = FileForAlignment_basename + WindowSuffix + '.fasta'
  AlignIO.write(SeqAlignmentHere, FileForAlnHere, 'fasta')

  # Infer the tree
  if Use_raxml_old:
    print('Running RAxML-old')
    NumTreesMade = pf.RunRAxMLOld(FileForAlnHere, TreeArgList, WindowSuffix,
    WindowAsStr, ChosenSeqStart, ChosenSeqEnd, TempFilesSet,)
    MLtreeFile = 'RAxML_bestTree.' + WindowSuffix + '.tree'
  elif Use_iqtree:
    print('Running IQtree')
    NumTreesMade = pf.RunIQtree(FileForAlnHere, TreeArgList, WindowSuffix, WindowAsStr,
                                ChosenSeqStart, ChosenSeqEnd)
    MLtreeFile = 'IQtree_' + WindowSuffix + '_.treefile'
  else:
    print('Running RAxML-NG')
    NumTreesMade = pf.RunRAxML(FileForAlnHere, TreeArgList, WindowSuffix, WindowAsStr,
                                ChosenSeqStart, ChosenSeqEnd, TempFilesSet)
    MLtreeFile = WindowSuffix + '.raxml.bestTree'

  if NumTreesMade != 1:
    print('Problem inferring a tree in window', str(ChosenSeqStart) + '-' + \
    str(ChosenSeqEnd) + '. Quitting', file=sys.stderr)
    exit(1)

  if not os.path.isfile(MLtreeFile):
    if Use_iqtree:
      print('Error: we lost the tree file produced by IQtree -', MLtreeFile + \
            '. Please report this to Chris Wymant. Quitting', file=sys.stderr)
      exit(1)
    else:
      print('Error: we lost the tree file produced by RAxML -', MLtreeFile + \
           '. Please report this to Chris Wymant. Quitting', file=sys.stderr)
      exit(1)

  proc = subprocess.Popen([TreeSizeCode, MLtreeFile], stdout=subprocess.PIPE,
  stderr=subprocess.PIPE)
  out, err = proc.communicate()
  TreeSizeCodeExitStatus = proc.returncode
  try:
    assert TreeSizeCodeExitStatus == 0
    TreeSize = float(out)
  except (AssertionError, ValueError):
    print('Problem calculating the size of the tree', MLtreeFile, 'using',
    TreeSizeCode + ':', err + '\nQuitting.', file=sys.stderr)
    raise

  return TreeSize

# Process all windows
if multithread:
  pool = Pool(args.threads)
  TreeSizes = pool.map(GetTreeSizeFromWindow, range(NumWindows))
else:
  TreeSizes = [GetTreeSizeFromWindow(i) for i in range(NumWindows)]

# Write the output file with tree sizes by window
with open(OutFileByWindow, 'w') as f:
  f.write('Window start,Window End,Median patristic pairwise distance\n')
  for WindowNumber in range(NumWindows):
    f.write(str(WindowStarts[WindowNumber]) + ',' + \
    str(WindowEnds[WindowNumber]) + ',' + str(TreeSizes[WindowNumber]) + '\n')

with open(OutFileByPosition, 'w') as f:
  proc = subprocess.Popen([ToPerPositionCode, OutFileByWindow], stdout=f,
  stderr=subprocess.PIPE)
  out, err = proc.communicate()
  ExitStatus = proc.returncode
  if ExitStatus != 0:
    print('Problem converting per-window tree sizes to per-position tree',
    'sizes using', ToPerPositionCode + ':', err + '\nQuitting.',
    file=sys.stderr)
    exit(1)

