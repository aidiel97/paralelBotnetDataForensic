import networkx as nx
import pandas as pd
from tqdm import tqdm
import time

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

def graphToTabular(G, raw_df):
    ctx = 'Graph based analysis - Graph to Tabular'
    start = watcherStart(ctx)

    raw_df['ActivityLabel'] = raw_df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
    botnet = raw_df['ActivityLabel'] == 1
    botnet_df = raw_df[botnet]
    normal = raw_df['ActivityLabel'] == 0
    normal_df = raw_df[normal]
    listBotnetAddress = botnet_df['SrcAddr'].unique()
    listNormalAddress = normal_df['SrcAddr'].unique()
    
    objData = {}
    totalNodes = G.number_of_nodes()
    progress_bar = tqdm(total=totalNodes, desc="Processing", unit="item")
    for node in G.nodes():
        label = 'botnet'
        activityLabel = 1
        if node not in listBotnetAddress:
            label = 'normal'
            activityLabel = 0
        
        out_data = raw_df[raw_df[['SrcAddr','Proto']].apply(tuple, axis=1).isin([node])]
        out_srcDesc = out_data['SrcBytes'].describe()
        in_data = raw_df[raw_df[['DstAddr','Proto']].apply(tuple, axis=1).isin([node])]
        in_srcDesc = in_data['SrcBytes'].describe()
        obj={
            'Address' : node[0],
            'Proto': node[1],

            'OutDegree': G.out_degree(node),
            'IntensityOutDegree': G.out_degree(node, weight='weight'),
            'SumSentBytes': out_data['SrcBytes'].sum(),
            'MeanSentBytes': out_srcDesc['mean'],
            'MedSentBytes': out_srcDesc['50%'],
            'CVSentBytes': (out_srcDesc['std']/out_srcDesc['mean'])*100,

            'InDegree': G.in_degree(node),
            'IntensityInDegree': G.in_degree(node, weight='weight'),
            'SumReceivedBytes': in_data['SrcBytes'].sum(),
            'MeanReceivedBytes': in_srcDesc['mean'],
            'MedReceivedBytes': in_srcDesc['50%'],
            'CVReceivedBytes': (in_srcDesc['std']/in_srcDesc['mean'])*100,

            'Label': label,
            'ActivityLabel': activityLabel
        }
        
        objData[node] = obj
        time.sleep(0.1)  # Simulate work with a delay

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Print a completion message
    print("Processing complete.")
    watcherEnd(ctx, start)
    return objData

def dftoGraph(datasetDetail):
    ctx = 'Graph based analysis - DF to Graph'
    start = watcherStart(ctx)
    raw_df = loader.binetflow(
        datasetDetail['datasetName'],
        datasetDetail['selected'],
        datasetDetail['stringDatasetName'])

    # keyFilter = ('147.32.84.165', 'udp')
    # new_df = botnet_df[botnet_df[['SrcAddr','Proto']].apply(tuple, axis=1).isin([keyFilter])]
    # print(new_df['SrcBytes'].sum())

    raw_df = raw_df.fillna('-')
    G = nx.DiGraph(directed=True)
    generatorWithEdgesArray(G, raw_df)
    objData = graphToTabular(G, raw_df)
    
    filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
    exportWithArrayOfObject(list(objData.values()), filename)

    watcherEnd(ctx, start)
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


    


