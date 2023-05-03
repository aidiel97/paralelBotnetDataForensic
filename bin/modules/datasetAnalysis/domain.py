import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing
import bin.modules.miner.sequenceMiner as minerTools
import bin.modules.machineLearning.domain as ml

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.dirManagement import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

botnetDiff = {}
backgroundDiff = {}
normalDiff = {}

def timeGap(): main('Diff')
def SrcBytes(): main('SrcBytes')

def main(feature):
  dataset = {
    'list': ctu,
    'name': 'CTU-13 (Local Source)',
    'shortName': 'ctu'
  }
  for scenario in dataset['list']:
    print(scenario)
    flow(dataset['list'], dataset['name'], dataset['shortName'], scenario, feature)

  # Create a boxplot
  plt.figure()
  botnetEquate = equateListLength(botnetDiff)
  botnetDf = pd.DataFrame(botnetEquate)
  botnetDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  # Adjust the spacing between the subplots to prevent the x-tick labels from being cropped
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('collections/'+feature+'-botnet-boxplot.png')
  botnetDf.describe().transpose().to_csv('collections/'+feature+'-botnet-describe.csv')

  plt.figure()
  backgroundEquate = equateListLength(backgroundDiff)
  backgroundDf = pd.DataFrame(backgroundEquate)
  backgroundDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  # Adjust the spacing between the subplots to prevent the x-tick labels from being cropped
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('collections/'+feature+'-background-boxplot.png')
  backgroundDf.describe().transpose().to_csv('collections/'+feature+'-background-describe.csv')

  plt.figure()
  normalEquate = equateListLength(normalDiff)
  normalDf = pd.DataFrame(normalEquate)
  normalDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  # Adjust the spacing between the subplots to prevent the x-tick labels from being cropped
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('collections/'+feature+'-normal-boxplot.png')
  normalDf.describe().transpose().to_csv('collections/'+feature+'-normal-describe.csv')

def flow(datasetName, stringDatasetName, shortName, selected, feature):
  ctx=feature+' Analysis with Statistical Approach'
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
  tgBotnet = botnet.loc[botnet[feature] != 0, feature].values.tolist()
  tgBackground = background.loc[background[feature] != 0, feature].values.tolist()
  tgNormal = normal.loc[normal[feature] != 0, feature].values.tolist()

  datasetVariableName = shortName+'('+selected[8:]+')'
  botnetDiff[datasetVariableName] = tgBotnet
  backgroundDiff[datasetVariableName] = tgBackground
  normalDiff[datasetVariableName] = tgNormal

  watcherEnd(ctx, start)

def equateListLength(dct):
  max_length = max(len(lst) for lst in dct.values())
  # Pad each list in x with zeros to make them all have the same length
  for key in dct:
      lst = dct[key]
      if len(lst) < max_length:
          lst += [np.nan] * (max_length - len(lst))
  
  return dct

def sequence():
  cv = lambda x: np.std(x) / np.mean(x)*100 #coefficient of variation (CV)
  sumOf = lambda x: np.sum(x)
  trainDataset = [
    'scenario3','scenario4','scenario5',
    'scenario7',
    'scenario10','scenario11','scenario12','scenario13'
    ]
  arrayDf = []
  datasetName = ctu
  stringDatasetName = 'ctu'
  for selected in trainDataset:
    df = loader.binetflow(datasetName, selected, stringDatasetName)

    df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
    botnet = df[df['ActivityLabel'] == 'botnet']
    normal = df[df['ActivityLabel'] == 'normal']
    background = df[df['ActivityLabel'] == 'background']

    all_df = minerTools.withDataframe(df)
    all_df['elementsInSequence'] = all_df.groupby('SequenceId')['SequenceId'].transform('count')

    botnet_df = minerTools.withDataframe(botnet)
    botnet_df['elementsInSequence'] = botnet_df.groupby('SequenceId')['SequenceId'].transform('count')
    botnet_df['SrcBytesCV'] = botnet_df.groupby('SequenceId')['SrcBytes'].transform(cv)
    botnet_df['SrcBytesSeq'] = botnet_df.groupby('SequenceId')['SrcBytes'].transform(sumOf)
    botnet_df['TotBytesCV'] = botnet_df.groupby('SequenceId')['TotBytes'].transform(cv)
    botnet_df['TotBytesSeq'] = botnet_df.groupby('SequenceId')['TotBytes'].transform(sumOf)
    botnet_df['TotPktsCV'] = botnet_df.groupby('SequenceId')['TotPkts'].transform(cv)
    botnet_df['TotPktsSeq'] = botnet_df.groupby('SequenceId')['TotPkts'].transform(sumOf)
    botnet_df.describe().transpose().to_csv('collections/'+stringDatasetName+selected+'-botnet-sequenceAnalysis.csv')

    background_df = minerTools.withDataframe(background)
    background_df['elementsInSequence'] = background_df.groupby('SequenceId')['SequenceId'].transform('count')
    background_df['SrcBytesCV'] = background_df.groupby('SequenceId')['SrcBytes'].transform(cv)
    background_df['SrcBytesSeq'] = background_df.groupby('SequenceId')['SrcBytes'].transform(sumOf)
    background_df['TotBytesCV'] = background_df.groupby('SequenceId')['TotBytes'].transform(cv)
    background_df['TotBytesSeq'] = background_df.groupby('SequenceId')['TotBytes'].transform(sumOf)
    background_df['TotPktsCV'] = background_df.groupby('SequenceId')['TotPkts'].transform(cv)
    background_df['TotPktsSeq'] = background_df.groupby('SequenceId')['TotPkts'].transform(sumOf)
    background_df.describe().transpose().to_csv('collections/'+stringDatasetName+selected+'-background-sequenceAnalysis.csv')

    normal_df = minerTools.withDataframe(normal)
    normal_df['elementsInSequence'] = normal_df.groupby('SequenceId')['SequenceId'].transform('count')
    normal_df['SrcBytesCV'] = normal_df.groupby('SequenceId')['SrcBytes'].transform(cv)
    normal_df['SrcBytesSeq'] = normal_df.groupby('SequenceId')['SrcBytes'].transform(sumOf)
    normal_df['TotBytesCV'] = normal_df.groupby('SequenceId')['TotBytes'].transform(cv)
    normal_df['TotBytesSeq'] = normal_df.groupby('SequenceId')['TotBytes'].transform(sumOf)
    normal_df['TotPktsCV'] = normal_df.groupby('SequenceId')['TotPkts'].transform(cv)
    normal_df['TotPktsSeq'] = normal_df.groupby('SequenceId')['TotPkts'].transform(sumOf)
    normal_df.describe().transpose().to_csv('collections/'+stringDatasetName+selected+'-normal-sequenceAnalysis.csv')

    data_dict = equateListLength({
      'botnet': botnet_df['elementsInSequence'].values.tolist(),
      'background': background_df['elementsInSequence'].values.tolist(),
      'normal': normal_df['elementsInSequence'].values.tolist(),
      'allData': all_df['elementsInSequence'].values.tolist()
    })

    new_df = pd.DataFrame(data_dict)
    plt.figure()
    new_df.boxplot(showfliers=False)
    plt.xticks(rotation=90)
    # Adjust the spacing between the subplots to prevent the x-tick labels from being cropped
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('collections/'+stringDatasetName+selected+'elementInsequenceAnalysis.png')
    new_df.describe().transpose().to_csv('collections/'+stringDatasetName+selected+'elementInsequenceAnalysis.csv')

def exportDataset():
  ctx = 'exportDataset'
  start = watcherStart(ctx)
  checkDir('collections/dataset')

  for dataset in listAvailableDatasets[:3]:
    exportDir = 'collections/dataset/'+dataset['name']
    checkDir(exportDir)
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
      
      raw_df = raw_df.reset_index(drop=True)
      raw_df.to_csv(exportDir+'/'+scenario+'.csv', index=False)
      print(raw_df)

      df = ml.preProcessingModule(raw_df) 
      df = df.reset_index(drop=True)
      df.drop(columns='Label', inplace=True)
      df.to_csv(exportDir+'/preProcessed-'+scenario+'.csv', index=False)
      print(df)

  watcherEnd(ctx, start)