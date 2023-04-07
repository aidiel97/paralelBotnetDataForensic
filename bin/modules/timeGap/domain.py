import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.csvGenerator import exportWithObject

import pandas as pd
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

  # plt.savefig('collections/test.png')
  tgBotnet = botnet.loc[botnet['Diff'] != 0, 'Diff'].values.tolist()
  tgBackground = background.loc[background['Diff'] != 0, 'Diff'].values.tolist()
  tgNormal = normal.loc[normal['Diff'] != 0, 'Diff'].values.tolist()

  # Pad the shorter list with NaN values to match the length of the longer list
  max_len = max(len(tgBotnet), len(tgBackground), len(tgNormal))
  tgBotnet += [np.nan] * (max_len - len(tgBotnet))
  tgBackground += [np.nan] * (max_len - len(tgBackground))
  tgNormal += [np.nan] * (max_len - len(tgNormal))

  # Create a pandas DataFrame with the two columns
  df = pd.DataFrame({'botnet':tgBotnet, 'background':tgBackground, 'normal': tgNormal})

  # Create a boxplot of the two columns
  plt.figure()
  df.boxplot(column=['botnet'], showfliers=False)
  plt.savefig('collections/'+stringDatasetName+'_'+selected+'-botnet-boxplot.png')
  df.boxplot(column=['background'], showfliers=False)
  plt.savefig('collections/'+stringDatasetName+'_'+selected+'-background-boxplot.png')
  df.boxplot(column=['normal'], showfliers=False)
  plt.savefig('collections/'+stringDatasetName+'_'+selected+'-normal-boxplot.png')

  # unique, counts = np.unique(botnet['Diff'], return_counts=True)
  # print(unique)
  # plt.scatter(x=unique,y=counts,c='red')
  
  # unique, counts = np.unique(background['Diff'], return_counts=True)
  # plt.scatter(x=unique,y=counts,c='blue')

  # unique, counts = np.unique(normal['Diff'], return_counts=True)
  # plt.scatter(x=unique,y=counts,c='green')

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
  for dataset in listAvailableDatasets[:3]:
    print(dataset['name'])
    for scenario in dataset['list']:
      print(scenario)
      flow(dataset['list'], dataset['name'], scenario)

  # datasetName = ctu
  # stringDatasetName = 'ctu'
  # selected = 'scenario7'
  # flow(datasetName, stringDatasetName, selected)