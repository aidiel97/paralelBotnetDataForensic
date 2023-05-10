import numpy as np

import bin.helpers.utilities.dataLoader as loader
import bin.modules.machineLearning.machineLearning as mlTools
import bin.modules.miner.sequenceMiner as tools

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *

timeGapValue = 13
seqWidth = 3600
collection = 'sequences'
itemsetCollection = 'itemsets'
sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr
commonPorts = ['22','23','25','110','143','220','465','993','995','125','137','139','445','3389','20','21','123']

def unique(list1):
  # insert the list to the set
  list_set = set(list1)
  # convert the set to the list
  unique_list = (list(list_set))
  return unique_list

def packetAnalysis(df):
  ctx='Method Evaluation'
  start = watcherStart(ctx)
  # SrcBytesThreshold = df['SrcBytes'].mean()
  # elementInsequencethreshold = df['elementsInSequence'].mean()
  elementInsequencethreshold = 49 #minimum
  SrcBytesThreshold = 62 #minimum
  totPktsCVThreshold = 753 #maksimum

  cv = lambda x: np.std(x) / np.mean(x)*100 #coefficient of variation (CV)
  sumOf = lambda x: np.sum(x)

  df['elementsInSequence'] = df.groupby('SequenceId')['SequenceId'].transform('count')
  df['SrcBytesCV'] = df.groupby('SequenceId')['SrcBytes'].transform(cv)
  df['SrcBytesSeq'] = df.groupby('SequenceId')['SrcBytes'].transform(sumOf)
  df['TotBytesCV'] = df.groupby('SequenceId')['TotBytes'].transform(cv)
  df['TotBytesSeq'] = df.groupby('SequenceId')['TotBytes'].transform(sumOf)
  df['TotPktsCV'] = df.groupby('SequenceId')['TotPkts'].transform(cv)
  df['TotPktsSeq'] = df.groupby('SequenceId')['TotPkts'].transform(sumOf)

  df = df[df['elementsInSequence'] > 1]

  #query filtering ini perlu dianalisis
  df = df[
      (df['SrcBytesSeq'] > SrcBytesThreshold) |
      (df['elementsInSequence'] > elementInsequencethreshold)
    ]

  df = df[df['TotPktsCV'] < totPktsCVThreshold]

  watcherEnd(ctx, start)
  return df

def methodEvaluation(dataset, actual_df, predicted_df, method='Proposed Sequence Pattern Miner'):
  ctx='Method Evaluation'
  start = watcherStart(ctx)
  addressPredictedAsBotnet = predicted_df['SrcAddr'].unique()

  actual_df['ActualClass'] = actual_df['Label'].str.contains('botnet', case=False, regex=True)
  result_df = actual_df.groupby('SrcAddr')['ActualClass'].apply(lambda x: x.mode()[0]).reset_index()
  result_df.columns = ['SrcAddr','ActualClass']
  result_df['PredictedClass'] = result_df['SrcAddr'].isin(addressPredictedAsBotnet)

  mlTools.evaluation(dataset, result_df['ActualClass'], result_df['PredictedClass'], method)
  watcherEnd(ctx, start)

def main():
  ctx='Sequential Pattern Mining (Main) - Single Dataset'
  start = watcherStart(ctx)
  ##### single subDataset
  datasetDetail={
    'datasetName': ctu,
    'stringDatasetName': 'ctu',
    'selected': 'scenario11'
  }
  ##### with input menu
  # datasetName, stringDatasetName, selected = datasetMenu.getData()
  ##### with input menu
  raw_df = loader.binetflow(
    datasetDetail['datasetName'],
    datasetDetail['selected'],
    datasetDetail['stringDatasetName'])

  print('\n'+datasetDetail['stringDatasetName'])
  print(datasetDetail['selected'])

  processed_df = raw_df[~raw_df['Dport'].isin(commonPorts)] #filter traffic use common ports
  processed_df = processed_df[~processed_df['Sport'].isin(commonPorts)] #filter traffic use common ports
  processed_df = processed_df[processed_df['TotPkts'] < processed_df['TotPkts'].mean()]

  new_df = tools.withDataframe(processed_df)
  new_df = packetAnalysis(new_df)

  datasetName = datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']
  methodEvaluation(datasetName, raw_df, new_df, 'Proposed Sequence Pattern Miner')
  ##### single subDataset

  watcherEnd(ctx, start)

def executeAllData():
  ctx='Sequential Pattern Mining (Main) - Execute All Data'
  start = watcherStart(ctx)

  ##### loop all dataset
  for dataset in listAvailableDatasets[:3]:
    print('\n'+dataset['name'])
    for scenario in dataset['list']:
      print(scenario)
      datasetDetail={
        'datasetName': dataset['list'],
        'stringDatasetName': dataset['name'],
        'selected': scenario
      }

      raw_df = loader.binetflow(
        datasetDetail['datasetName'],
        datasetDetail['selected'],
        datasetDetail['stringDatasetName'])

      processed_df = raw_df[~raw_df['Dport'].isin(commonPorts)] #filter traffic use common ports
      processed_df = processed_df[~processed_df['Sport'].isin(commonPorts)] #filter traffic use common ports
      processed_df = processed_df[processed_df['TotPkts'] < processed_df['TotPkts'].mean()]

      new_df = tools.withDataframe(processed_df)
      new_df = packetAnalysis(new_df)

      datasetName = datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']
      methodEvaluation(datasetName, raw_df, new_df, 'Proposed Sequence Pattern Miner')
  ##### loop all dataset

  watcherEnd(ctx, start)
