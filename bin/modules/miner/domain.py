import uuid
import math
import pandas as pd
import numpy as np

import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing
import bin.modules.miner.model as model
import bin.modules.machineLearning.domain as ml
import bin.modules.machineLearning.machineLearning as mlTools
import bin.modules.utilities.domain as utilities
import bin.modules.miner.sequenceMiner as tools

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *
from itertools import combinations

timeGapValue = 80
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

def packetAnalysis(df, SrcBytesThreshold, SrcBytesCVThreshold):
  ctx='Method Evaluation'
  start = watcherStart(ctx)

  cv = lambda x: np.std(x) / np.mean(x)*100 #coefficient of variation (CV)
  sumOf = lambda x: np.sum(x)

  df['elementsInSequence'] = df.groupby('SequenceId')['SequenceId'].transform('count')
  df = df[df['elementsInSequence'] > 1 ]
  df['SrcBytesCV'] = df.groupby('SequenceId')['SrcBytes'].transform(cv)
  df['SrcBytesSeq'] = df.groupby('SequenceId')['SrcBytes'].transform(sumOf)
  df['TotBytesCV'] = df.groupby('SequenceId')['TotBytes'].transform(cv)
  df['TotBytesSeq'] = df.groupby('SequenceId')['TotBytes'].transform(sumOf)
  df['TotPktsCV'] = df.groupby('SequenceId')['TotPkts'].transform(cv)
  df['TotPktsSeq'] = df.groupby('SequenceId')['TotPkts'].transform(sumOf)
  df = df[df['SrcBytesCV'] <= SrcBytesCVThreshold ]
  df = df[df['SrcBytesSeq'] > SrcBytesThreshold]

  watcherEnd(ctx, start)
  return df

def methodEvaluation(dataset, actual_df, predicted_df):
  ctx='Method Evaluation'
  start = watcherStart(ctx)
  addressPredictedAsBotnet = predicted_df['SrcAddr'].unique()

  actual_df['ActualClass'] = actual_df['Label'].str.contains('botnet', case=False, regex=True)
  result_df = actual_df.groupby('SrcAddr')['ActualClass'].apply(lambda x: x.mode()[0]).reset_index()
  result_df.columns = ['SrcAddr','ActualClass']
  result_df['PredictedClass'] = result_df['SrcAddr'].isin(addressPredictedAsBotnet)

  mlTools.evaluation(dataset, result_df['ActualClass'], result_df['PredictedClass'], 'Proposed Sequence Pattern Miner')

  watcherEnd(ctx, start)

def main():
  ctx='Sequential Pattern Mining (Main)'
  start = watcherStart(ctx)
    ##### with input menu
  # datasetName, stringDatasetName, selected = datasetMenu.getData()
    ##### with input menu

    ##### single subDataset
  datasetDetail={
    'datasetName': ctu,
    'stringDatasetName': 'ctu',
    'selected': 'scenario4'
  }
  raw_df = loader.binetflow(
    datasetDetail['datasetName'],
    datasetDetail['selected'],
    datasetDetail['stringDatasetName'])

  df = raw_df.copy() #get a copy from dataset to prevent processed data
  result = ml.predict(df)
  raw_df['predictionResult'] = result
  processed_df = raw_df[~raw_df['Dport'].isin(commonPorts)] #filter traffic use common ports
  processed_df = processed_df[~processed_df['Sport'].isin(commonPorts)] #filter traffic use common ports
  processed_df = processed_df[processed_df['predictionResult'] == 0] #remove background (ActivityLabel == 1)

  # new_df = tools.withMongo(datasetDetail, processed_df)
  new_df = tools.withDataframe(processed_df)
  SrcBytesThreshold = raw_df['SrcBytes'].mean()
  SrcBytesCVThreshold = 75
  new_df = packetAnalysis(new_df, SrcBytesThreshold, SrcBytesCVThreshold)
  datasetName = datasetDetail.stringDatasetName+'-'+datasetDetail.selected
  methodEvaluation(datasetName, raw_df, new_df)

  # for segment in range(segmentLen):
  #   supportCounter(datasetDetail, itemsets, segment)
  #   combinationItem = combination(itemsets, segment)
  #   countCombinationSupport(datasetDetail, combinationItem, segment)
    ##### single subDataset

  #   ##### loop all dataset
  # for dataset in listAvailableDatasets[:3]:
  #   print(dataset['name'])
  #   for scenario in dataset['list']:
  #     print(scenario)
  #     datasetDetail={
  #       'datasetName': dataset['list'],
  #       'stringDatasetName': dataset['name'],
  #       'selected': scenario
  #     }

  #     df = loader.binetflow(
  #       datasetDetail['datasetName'],
  #       datasetDetail['selected'],
  #       datasetDetail['stringDatasetName'])
  #     df = raw_df.copy() #get a copy from dataset to prevent processed data
  #     result = ml.predict(df)
  #     raw_df['predictionResult'] = result
  #     processed_df = raw_df[raw_df['predictionResult'] == 0] #remove background (ActivityLabel == 1)
  #     itemsets = sequenceMiner(datasetDetail, processed_df)
  #     supportCounter(datasetDetail, itemsets)
  #   ##### loop all dataset

  watcherEnd(ctx, start)
