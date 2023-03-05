import bin.interfaces.cli.dataset as datasetMenu
import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as transform
import bin.modules.preProcessing.handlingNull as null
import bin.modules.preProcessing.cleansing as cleansing
import bin.modules.machineLearning as ml
from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *

import pandas as pd

def sensorBasedCausalityAnalysis():
  ctx = 'SensorBasedBotnetCausalityAnalysis'
  start = watcherStart(ctx)
  datasetName, stringDatasetName, selected = datasetMenu.getData()

  # datasetName = ncc2
  # stringDatasetName = 'ncc2'
  # selected = 'scenario1'

  df = loader.binetflow(datasetName, selected, stringDatasetName)

  ##### pre-processing
  #make new label for bot prediciton(1/0)
  df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  #transform with dictionary
  df['State']= df['State'].map(stateDict).fillna(0.0).astype(int)
  df['Proto']= df['Proto'].map(protoDict).fillna(0.0).astype(int)
  df.dropna(subset = ["DstAddr"], inplace=True)
  df.dropna(subset = ["SrcAddr"], inplace=True)
  df['Sport'] = pd.factorize(df.Sport)[0]
  df['Dport'] = pd.factorize(df.Dport)[0]
  #transform ip to integer
  df['SrcAddr'] = df['SrcAddr'].apply(transform.ipToInteger).fillna(0)
  df['DstAddr'] = df['DstAddr'].apply(transform.ipToInteger).fillna(0)
  

  null.setEmptyString(df)
  cleansing.featureDropping(df, ['sTos','dTos'])
  ##### pre-processing

  train, test = loader.splitTestAllDataframe(df)
  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]

  x_train=train.drop(categorical_features,axis=1)
  x_test=test.drop(categorical_features,axis=1)
  y_train = train['ActivityLabel']
  y_test = test['ActivityLabel']

  ml.modelling(x_train, y_train)
  # predictionResult = ml.classification(x_test)
  # ml.evaluation(ctx, y_test, predictionResult)

  watcherEnd(ctx, start)