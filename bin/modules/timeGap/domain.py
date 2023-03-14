import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.csvGenerator import exportWithObject

import os
import json

def flow(datasetName, stringDatasetName, selected):
  ctx='Sequential Pattern Mining for Detection'
  start = watcherStart(ctx)
  sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'StartTime', 'ActivityLabel'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None) #calculate diff with before event, negative convert to 0
  
  # print(df)
  # df[['StartTime','SrcAddr','Sport','DstAddr', 'Dport', 'Proto','ActivityLabel','Diff', 'Label']].to_csv('collections/df.csv',index=False)

  botnet = df[df['ActivityLabel'] == 'botnet']
  normal = df[df['ActivityLabel'] == 'normal']
  background = df[df['ActivityLabel'] == 'background']

  try:
    os.makedirs('collections/timeGap/'+stringDatasetName+'/'+selected)
  except FileExistsError:
    # directory already exists
    pass
  
  try:
    os.makedirs('collections/timeGap//'+stringDatasetName+'/'+selected)
  except FileExistsError:
    # directory already exists
    pass
  
  inclFeatures = ['Diff','SrcAddr','Sport','DstAddr', 'Dport', 'Proto']
  
  b_stat = json.loads(botnet[inclFeatures].describe().loc[['mean','std','min','max']].to_json())['Diff']
  b_stat['dataset'] = stringDatasetName
  b_stat['subDataset'] = selected
  b_stat['desc'] = 'botnet'

  print(b_stat)
  # botnet[inclFeatures].describe().to_csv('collections/timeGap/'+stringDatasetName+'/'+selected+'/botnet.csv',index=True)
  # normal[inclFeatures].describe().to_csv('collections/timeGap/'+stringDatasetName+'/'+selected+'/normal.csv',index=True)
  # background[inclFeatures].describe().to_csv('collections/timeGap/'+stringDatasetName+'/'+selected+'/background.csv',index=True)

  watcherEnd(ctx, start)

def main():
  # for dataset in listAvailableDatasets[:3]:
  #   print(dataset['name'])
  #   for scenario in dataset['list']:
  #     print(scenario)
  #     flow(dataset['list'], dataset['name'], scenario)

  datasetName = ctu
  stringDatasetName = 'ctu'
  selected = 'scenario7'
  flow(datasetName, stringDatasetName, selected)