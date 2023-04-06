import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing
import bin.modules.miner.model as model

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *


def main():
  ctx='Sequential Pattern Mining for Detection'
  start = watcherStart(ctx)

  sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

  # datasetName, stringDatasetName, selected = datasetMenu.getData()

  datasetName = ctu
  stringDatasetName = 'ctu'
  selected = 'scenario7'

  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df = df.sort_values(by=[sequenceOf, 'StartTime'])
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df['Diff'] = df['Unix'] - df['Unix'][0]
  df['NewLabel'] = df['Label'].apply(preProcessing.labelProcessing)

  ipInData = list(df['SrcAddr'].unique()) + list(df['DstAddr'].unique())
  # print(unique(ipInData))
  # print(df['NewLabel'].unique())
  # print(df['Label'].unique())

  dictLabel = model.main(model.labelModel, df['Label'].unique())
  print(dictLabel)
  insertOne({
    'ii':1
  },'laaa')
  # upsertmany((list(model.labelModel.keys())[0]),dictLabel,'label')

  exit()
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

def unique(list1):
  # insert the list to the set
  list_set = set(list1)
  # convert the set to the list
  unique_list = (list(list_set))
  return unique_list
