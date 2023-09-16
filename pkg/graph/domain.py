import networkx as nx
import pandas as pd

import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as pp
import pkg.machineLearning.machineLearning as ml

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.dataLoader import splitTestAllDataframe
from helpers.utilities.database import *
from helpers.utilities.csvGenerator import exportWithArrayOfObject

from pkg.graph.models import *
from pkg.graph.generator import *

def dftoGraph(datasetDetail):
    raw_df = loader.binetflow(
        datasetDetail['datasetName'],
        datasetDetail['selected'],
        datasetDetail['stringDatasetName'])
    # # datasetInSpecificBotnet = ['147.32.84.165','147.32.84.191','147.32.84.192']
    raw_df['Label'] = raw_df['Label'].apply(pp.labelSimplier)
    botnet = raw_df[raw_df['Label'] == 'botnet']
    normal = raw_df.loc[raw_df['Label'] == 'normal']
    # normal = raw_df.loc[(raw_df['Label'] == 'normal') & (raw_df['DstAddr'].isin(datasetInSpecificBotnet))]
    # others = raw_df[raw_df['DstAddr'].isin(datasetInSpecificBotnet)]
    # withoutBackground = raw_df[raw_df['Label'] != 'background']
    listBotnetAddress = botnet['SrcAddr'].unique()
    # print(listBotnetAddress)
    listNormalAddress = normal['SrcAddr'].unique()

    G = nx.DiGraph(directed=True)
    generatorWithEdgesArray(G, raw_df)
    
    NG = nx.DiGraph()
    generatorWithEdgesArray(NG, raw_df, True)
    
    meanG = nx.DiGraph()
    generatorWithEdgesArray(meanG, raw_df, True, 'mean')
    
    medG = nx.DiGraph()
    generatorWithEdgesArray(medG, raw_df, True, 'mean')

    CVG = nx.DiGraph()
    generatorWithEdgesArray(CVG, raw_df, True, 'cv')

    objData = {}
    for node in G.nodes():
        label = 'botnet'
        activityLabel = 1
        if node not in listBotnetAddress:
            label = 'normal'
            activityLabel = 0
        obj = {
            'Address': node,
            'WeightedOutDegree': G.out_degree(node, weight='weight'),
            'WeightedInDegree': G.in_degree(node, weight='weight'),
            'OutDegree': G.out_degree(node),
            'InDegree': G.in_degree(node),
            'TotSentBytes': NG.out_degree(node, weight='weight'),
            'TotReceivedBytes': NG.in_degree(node, weight='weight'),
            'meanSentBytes': meanG.out_degree(node, weight='weight'),
            'meanReceivedBytes': meanG.in_degree(node, weight='weight'),
            'medSentBytes': medG.out_degree(node, weight='weight'),
            'medReceivedBytes': medG.in_degree(node, weight='weight'),
            'CVSentBytes': CVG.out_degree(node, weight='weight'),
            'CVReceivedBytes': CVG.in_degree(node, weight='weight'),
            'Label': label,
            'ActivityLabel': activityLabel
        }
        objData[node] = obj
    
    filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
    exportWithArrayOfObject(list(objData.values()), filename)

    return objData


def singleData():
    ctx = 'Graph based analysis - Single Dataset'
    start = watcherStart(ctx)
    ##### single subDataset
    datasetDetail = {
        'datasetName': ctu,
        'stringDatasetName': 'ctu',
        'selected': 'scenario7'
    }
    ##### with input menu
    # datasetName, stringDatasetName, selected = datasetMenu.getData()
    ##### with input menu
    
    objData = dftoGraph(datasetDetail)

    df = pd.DataFrame(objData.values())
    df = df.fillna(0)
    train, test = splitTestAllDataframe(df,0.7)
    x_train = train.drop(['Label','ActivityLabel', 'Address'],axis=1)
    y_train = train['ActivityLabel']
    
    x_test = test.drop(['Label','ActivityLabel', 'Address'],axis=1)
    y_test = test['ActivityLabel']
    
    ml.modelling(x_train, y_train, 'knn')
    predictionResult = ml.classification(x_test, 'knn')
    print(predictionResult)
    ml.evaluation(ctx, y_test, predictionResult, 'knn')

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

        df = pd.DataFrame(objData.values())
        df = df.fillna(0)
        train, test = splitTestAllDataframe(df,0.7)
        x_train = train.drop(['ActivityLabel', 'Address'],axis=1)
        y_train = train['ActivityLabel']

        x_test = test.drop(['ActivityLabel', 'Address'],axis=1)
        y_test = test['ActivityLabel']
        
        ml.modelling(x_train, y_train, 'knn')
        predictionResult = ml.classification(x_test, 'knn')
        print(predictionResult)
        ml.evaluation(ctx, y_test, predictionResult, 'knn')

  ##### loop all dataset

  watcherEnd(ctx, start)


    


