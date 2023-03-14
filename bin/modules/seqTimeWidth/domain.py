import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *

def main():
  ctx='Sequential Pattern Mining for Detection'
  start = watcherStart(ctx)

  sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

  # datasetName, stringDatasetName, selected = datasetMenu.getData()

  datasetName = ctu
  stringDatasetName = 'ctu'
  selected = 'scenario4'

  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df = df.sort_values(by=[sequenceOf, 'StartTime'])
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df['Diff'] = df['Unix'] - df['Unix'][0]

  seq = [] #in one subDataset has one sequence
  subSeq = [] #subSequence is based on SrcAddr
  element = []
  totSeqPatternTime = 0
  existIP = ''
  netT = ()
  #sequence pattern mining
  for index, row in df.iterrows():
    netT = (
      row['SrcAddr'],
      row['DstAddr'],
      # row['Sport'],row['Dport'],row['State'],
      # row['TotPkts'],row['TotBytes'],row['SrcBytes'],row['Diff']
    )
    if(existIP == '' or existIP == row[sequenceOf]):
      totSeqPatternTime += row['Diff']
      #subSeq is created from collection of NetT which in same time Window (1hrs)
      if(totSeqPatternTime < 3600):
        element.append(netT)
      else:
        subSeq.append(element)
        element = [netT]
        totSeqPatternTime = 0
    else:
      seq.append(subSeq)
      subSeq = [element]
    
    existIP = row[sequenceOf]

  supportCount = {}
  #frequent analysis
  for subSeq in seq:
    for element in subSeq:
      if(tuple(element) not in supportCount):
        supportCount[tuple(element)] = 1
      else:
        supportCount[tuple(element)] += 1
  
  sortedBySupport = sorted(supportCount.items(), key=lambda x:x[1], reverse=True)
  # print(supportCount)
  print(sortedBySupport[:2])
  watcherEnd(ctx, start)