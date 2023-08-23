import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as transform
import pkg.preProcessing.handlingNull as null
import pkg.preProcessing.cleansing as cleansing
import pkg.machineLearning.machineLearning as ml
import pkg.miner.domain as miner

from helpers.utilities.watcher import *
from helpers.common.main import *

import pandas as pd

def preProcessingModule(df):
  #make new label for background prediciton(1/0)
  df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  #make new label for background prediciton(1/0)

  #transform with dictionary
  df['State']= df['State'].map(stateDict).fillna(0.0).astype(int)
  df['Proto']= df['Proto'].map(protoDict).fillna(0.0).astype(int)
  #transform with dictionary

  df['StartTime'] = df['StartTime'].apply(transform.timeToUnix).fillna(0)

  df['Sport'] = pd.factorize(df.Sport)[0]
  df['Dport'] = pd.factorize(df.Dport)[0]

  #transform ip to integer
  df.dropna(subset = ["DstAddr"], inplace=True)
  df.dropna(subset = ["SrcAddr"], inplace=True)
  df['SrcAddr'] = df['SrcAddr'].apply(transform.ipToInteger).fillna(0)
  df['DstAddr'] = df['DstAddr'].apply(transform.ipToInteger).fillna(0)
  #transform ip to integer

  null.setEmptyString(df)
  # cleansing.featureDropping(df, ['sTos','dTos','Dir'])

  #one hot encode
  dir_values_to_encode = ['  <->','   ->','  who','  <-','  <?>','   ?>','  <?']
  dummy_cols =pd.get_dummies(
    df['Dir'].apply(lambda x: x if x in dir_values_to_encode else 'other'), columns=dir_values_to_encode, prefix='Dir')
  df = pd.concat([df,dummy_cols],axis=1)
  df.drop(columns='Dir', axis=1, inplace=True)
  
  return df

def predict(df, algorithm='randomForest'):
  ctx = 'Machine Learning Classification'
  start = watcherStart(ctx)

  df = preProcessingModule(df) # pre-processing
  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]
  x = df.drop(categorical_features,axis=1)
  y = df['ActivityLabel']
  predictionResult = ml.classification(x, algorithm)
  ml.evaluation(ctx, y, predictionResult, algorithm)
  
  watcherEnd(ctx, start)
  return predictionResult

def modellingWithCTU(algorithm='randomForest'):
  ctx = 'Modelling with CTU dataset'
  start = watcherStart(ctx)

  #modelling
  #### PRE DEFINED TRAINING DATASET FROM http://dx.doi.org/10.1016/j.cose.2014.05.011
  trainDataset = ['scenario3','scenario4','scenario5','scenario7','scenario10','scenario11','scenario12','scenario13']
  arrayDf = []
  datasetName = ctu
  stringDatasetName = 'ctu'
  for selected in trainDataset:
    arrayDf.append(loader.binetflow(datasetName, selected, stringDatasetName))
  df = pd.concat(arrayDf, axis=0)
  df.reset_index(drop=True, inplace=True)
  df = preProcessingModule(df) # pre-processing
  #### PRE DEFINED TRAINING DATASET FROM http://dx.doi.org/10.1016/j.cose.2014.05.011

  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]
  x = df.drop(categorical_features,axis=1)
  y = df['ActivityLabel']
  ml.modelling(x, y, algorithm)
  #modelling

  #evaluate
  datasetName = ctu
  stringDatasetName = 'ctu'
  selected = 'scenario7'
  test_df = loader.binetflow(datasetName, selected, stringDatasetName)
  predict(test_df, algorithm)
  #evaluate

  watcherEnd(ctx, start)

def executeAllData():
  ctx='Machine learning Classification - Execute All Data'
  start = watcherStart(ctx)

  for algo in list(ml.algorithmDict.keys()):
    modellingWithCTU(algo)
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

        df = raw_df.copy() #get a copy from dataset to prevent processed data
        result = predict(df, algo)
        raw_df['predictionResult'] = result
        new_df = raw_df[raw_df['predictionResult'] == 1]
        
        datasetName = datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']
        miner.methodEvaluation(datasetName, raw_df, new_df, algo)
  ##### loop all dataset

  watcherEnd(ctx, start)

def singleData():
  ctx='Machine learning Classification - Single Data'
  start = watcherStart(ctx)

  for algo in list(ml.algorithmDict.keys()):
    modellingWithCTU(algo)
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

    df = raw_df.copy() #get a copy from dataset to prevent processed data
    result = predict(df, algo)
    raw_df['predictionResult'] = result
    new_df = raw_df[raw_df['predictionResult'] == 1]
    
    datasetName = datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']
    miner.methodEvaluation(datasetName, raw_df, new_df, algo)
