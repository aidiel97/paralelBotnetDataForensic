import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.csvGenerator import exportWithObject

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def flow(datasetName, stringDatasetName, selected):
  ctx='Sequential Pattern Mining for Detection'
  start = watcherStart(ctx)
  sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'StartTime', 'ActivityLabel'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None) #calculate diff with before event, negative convert to 0

  botnet = df[df['ActivityLabel'] == 'botnet']
  normal = df[df['ActivityLabel'] == 'normal']
  background = df[df['ActivityLabel'] == 'background']

  print(botnet['SrcAddr'].value_counts())

  # unique, counts = np.unique(botnet['Diff'], return_counts=True)
  # plt.scatter(x=unique,y=counts,c='red')
  
  # unique, counts = np.unique(background['Diff'], return_counts=True)
  # plt.scatter(x=unique,y=counts,c='blue')

  # unique, counts = np.unique(normal['Diff'], return_counts=True)
  # plt.scatter(x=unique,y=counts,c='green')

  # plt.savefig('collections/test.png')
  # exit()

  # inclFeatures = ['Diff','SrcAddr','Sport','DstAddr', 'Dport', 'Proto']

  # backgorund_stat = json.loads(background[inclFeatures].describe().loc[['mean','std','min','max']].to_json())['Diff']
  # backgorund_stat['dataset'] = stringDatasetName
  # backgorund_stat['subDataset'] = selected
  # backgorund_stat['desc'] = 'background'  
  # exportWithObject(backgorund_stat,'collections/timeGap.csv')

  # botnet_stat = json.loads(botnet[inclFeatures].describe().loc[['mean','std','min','max']].to_json())['Diff']
  # botnet_stat['dataset'] = stringDatasetName
  # botnet_stat['subDataset'] = selected
  # botnet_stat['desc'] = 'botnet'  
  # exportWithObject(botnet_stat,'collections/timeGap.csv')

  # normal_stat = json.loads(normal[inclFeatures].describe().loc[['mean','std','min','max']].to_json())['Diff']
  # normal_stat['dataset'] = stringDatasetName
  # normal_stat['subDataset'] = selected
  # normal_stat['desc'] = 'normal'  
  # exportWithObject(normal_stat,'collections/timeGap.csv')



  watcherEnd(ctx, start)

def main():
  # for dataset in listAvailableDatasets[:3]:
  #   print(dataset['name'])
  #   for scenario in dataset['list']:
  #     print(scenario)
  #     flow(dataset['list'], dataset['name'], scenario)

  datasetName = ncc2
  stringDatasetName = 'ncc2'
  selected = 'scenario2'
  flow(datasetName, stringDatasetName, selected)