import networkx as nx
import pandas as pd
from tqdm import tqdm
import time

import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as pp
import pkg.machineLearning.machineLearning as ml
import interfaces.cli.dataset as datasetMenu

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.dataLoader import splitTestAllDataframe
from helpers.utilities.database import *
from helpers.utilities.csvGenerator import exportWithArrayOfObject
from pkg.graph.models import *
from pkg.graph.generator import *
from pkg.graph.handler import *

from sklearn.preprocessing import StandardScaler 

def graphClassificationModelling():
    ctx = 'Graph based classification - Modelling'
    start = watcherStart(ctx)

    keysAlg = list(ml.algorithmDict.keys())
    print("Choose one of this algorithm to train :")
    
    i=1
    for alg in keysAlg:
        print(str(i)+". "+alg)
        i+=1
    
    indexAlg = input("Enter Menu: ")
    algorithm = keysAlg[int(indexAlg)-1]

    #modelling
    #### PRE DEFINED TRAINING DATASET FROM http://dx.doi.org/10.1016/j.cose.2014.05.011
    trainDataset = ['scenario3','scenario4','scenario5','scenario7','scenario10','scenario11','scenario12','scenario13']
    arrayDf = []
    datasetName = nccGraphCTU
    stringDatasetName = 'nccGraphCTU'
    for selected in trainDataset:
        arrayDf.append(loader.binetflow(datasetName, selected, stringDatasetName))
    df = pd.concat(arrayDf, axis=0)
    df.reset_index(drop=True, inplace=True)
    botIP = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    df['CVReceivedBytes'] = df['CVReceivedBytes'].fillna(0)
    df['CVSentBytes'] = df['CVSentBytes'].fillna(0)
    df['ActivityLabel'] = df['Address'].isin(botIP).astype(int)
    
    categorical_features=[feature for feature in df.columns if (
        df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
    )]
    y = df['ActivityLabel']
    x = df.drop(categorical_features,axis=1)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # from sklearn.decomposition import PCA
    # pca = PCA().fit(x)
    # plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
    # plt.title('Cumulative explained variance by number of principal components', size=20)
    # plt.savefig('collections/pca.png', bbox_inches='tight')
    
    # loadings = pd.DataFrame(
    #     data=pca.components_.T * np.sqrt(pca.explained_variance_), 
    #     columns=[f'PC{i}' for i in range(1, len(x.columns) + 1)],
    #     index=x.columns
    # )
    # loadings.head()
    # pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    # pc1_loadings = pc1_loadings.reset_index()
    # pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

    # plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
    # plt.title('PCA loading scores (first principal component)', size=20)
    # plt.xticks(rotation='vertical')
    # plt.savefig('collections/pca-loadingScores.png', bbox_inches='tight')

    ml.modelling(x, y, algorithm)
    
    test_df = loader.binetflow(nccGraphCTU, 'scenario9', 'nccGraphCTU')
    test_df['ActivityLabel'] = test_df['Address'].isin(botIP).astype(int)
    
    for col in protoDict.keys():
        test_df[col] = (test_df['Proto'] == col).astype(int)

    test_df['CVReceivedBytes'] = test_df['CVReceivedBytes'].fillna(0)
    test_df['CVSentBytes'] = test_df['CVSentBytes'].fillna(0)

    y_test = test_df['ActivityLabel']
    x_test = test_df.drop(categorical_features,axis=1)
    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    predict_result = ml.classification(x_test, algorithm)
    ml.evaluation(ctx, y_test, predict_result, algorithm)

    watcherEnd(ctx, start)

def singleData():
    ctx = 'Graph based analysis - Single Dataset'
    start = watcherStart(ctx)
    ##### single subDataset
    # datasetDetail = {
    #     'datasetName': ncc2,
    #     'stringDatasetName': 'ncc2',
    #     'selected': 'scenario3'
    # }
    ##### with input menu
    datasetName, stringDatasetName, selected = datasetMenu.getData()
    ##### with input menu
    
    datasetDetail = {
        'datasetName': datasetName,
        'stringDatasetName': stringDatasetName,
        'selected': selected
    }
    
    dftoGraph(datasetDetail)

    # df = pd.DataFrame(objData.values())
    # df = df.fillna(0)
    # train, test = splitTestAllDataframe(df,0.7)
    # x_train = train.drop(['Label','ActivityLabel', 'Address'],axis=1)
    # y_train = train['ActivityLabel']
    
    # x_test = test.drop(['Label','ActivityLabel', 'Address'],axis=1)
    # y_test = test['ActivityLabel']
    
    # ml.modelling(x_train, y_train, 'knn')
    # predictionResult = ml.classification(x_test, 'knn')
    # print(predictionResult)
    # ml.evaluation(ctx, y_test, predictionResult, 'knn')

    # ##### visualize
    # exportGraph(G, 'collections/graph.png')
    # exportGraph(NG, 'collections/weighted-graph.png')
    # ##### visualize

    watcherEnd(ctx, start)
    
def executeAllData():
  ctx='Graph based analysis - Execute All Data'
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

        objData = dftoGraph(datasetDetail)

        # df = pd.DataFrame(objData.values())
        # df = df.fillna(0)
        # train, test = splitTestAllDataframe(df,0.7)
        # x_train = train.drop(['ActivityLabel', 'Address'],axis=1)
        # y_train = train['ActivityLabel']

        # x_test = test.drop(['ActivityLabel', 'Address'],axis=1)
        # y_test = test['ActivityLabel']
        
        # ml.modelling(x_train, y_train, 'knn')
        # predictionResult = ml.classification(x_test, 'knn')
        # print(predictionResult)
        # ml.evaluation(ctx, y_test, predictionResult, 'knn')

  ##### loop all dataset

  watcherEnd(ctx, start)
