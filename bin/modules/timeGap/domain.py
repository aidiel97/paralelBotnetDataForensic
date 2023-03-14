import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *

import os

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
    os.makedirs('collections/'+stringDatasetName+'/'+selected)
  except FileExistsError:
    # directory already exists
    pass
  
  try:
    os.makedirs('collections/'+stringDatasetName+'/'+selected)
  except FileExistsError:
    # directory already exists
    pass
  
  inclFeatures = ['Diff','SrcAddr','Sport','DstAddr', 'Dport', 'Proto']
  botnet[inclFeatures].describe().to_csv('collections/'+stringDatasetName+'/'+selected+'/botnet.csv',index=True)
  normal[inclFeatures].describe().to_csv('collections/'+stringDatasetName+'/'+selected+'/normal.csv',index=True)
  background[inclFeatures].describe().to_csv('collections/'+stringDatasetName+'/'+selected+'/background.csv',index=True)

  watcherEnd(ctx, start)

def main():
  for dataset in listAvailableDatasets[:3]:
    print(dataset['name'])
    for scenario in dataset['list']:
      print(scenario)
      flow(dataset['list'], dataset['name'], scenario)