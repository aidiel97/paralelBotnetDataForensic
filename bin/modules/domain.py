import bin.interfaces.cli.dataset as datasetMenu
import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as transform
import bin.modules.preProcessing.handlingNull as null
import bin.modules.preProcessing.cleansing as cleansing
import bin.modules.machineLearning as ml
from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def preProcessing(df):
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

  return df

def sensorBasedCausalityAnalysis():
  ctx = 'SensorBasedBotnetCausalityAnalysis'
  start = watcherStart(ctx)
  # datasetName, stringDatasetName, selected = datasetMenu.getData()

  datasetName = ncc2
  stringDatasetName = 'ncc2'

  selected = 'scenario1'
  raw_df = loader.binetflow(datasetName, selected, stringDatasetName)
  df = raw_df
  df = preProcessing(df)

  train, test = loader.splitTestAllDataframe(df)
  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]

  x_train=train.drop(categorical_features,axis=1)
  x_test=test.drop(categorical_features,axis=1)
  y_train = train['ActivityLabel']
  y_test = test['ActivityLabel']

  # ml.modelling(x_train, y_train)
  predictionResult = ml.classification(x_test)
  ml.evaluation(ctx, y_test, predictionResult)
  
  dropFeature = [feature for feature in raw_df.columns if (feature =='SensorId' or feature =='ActivityLabel')]
  predictedDf = x_test.assign(PredictedLabel=predictionResult)
  
  normal_df=predictedDf[predictedDf['PredictedLabel'].isin([0])]
  bot_df=predictedDf[predictedDf['PredictedLabel'].isin([1])].drop(['PredictedLabel'], axis=1)
  print(bot_df.to_numpy())

  s2_df = preProcessing(loader.binetflow(datasetName, 'scenario2', stringDatasetName)).drop(categorical_features,axis=1)
  print(s2_df.to_numpy())
  # s3_df = preProcessing(loader.binetflow(datasetName, 'scenario3', stringDatasetName)).drop(categorical_features,axis=1)
  # print(s3_df.to_numpy())

  for row in s2_df.to_numpy():
    similarity = cosine_similarity([bot_df.to_numpy()[0]], [row])
    print(similarity)

  watcherEnd(ctx, start)