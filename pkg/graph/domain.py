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
    listBotnetAddress = botnet_df['SrcAddr'].unique()

    # normal = raw_df['ActivityLabel'] == 0
    # normal_df = raw_df[normal]
    # listNormalAddress = normal_df['SrcAddr'].unique()
    
    result_src_df = raw_df.groupby(['SrcAddr', 'Proto'])['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_dst_df = raw_df.groupby(['DstAddr', 'Proto'])['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_src_df = result_src_df.fillna(0)
    result_dst_df = result_dst_df.fillna(0)
    # result_df.columns = ['SrcAddr', 'Proto', 'mean', 'SrcBytes_std', 'SrcBytes_median', 'SrcBytes_sum']
    # print(result_df)
    # specific_row = result_df.loc[(result_df['SrcAddr'] == '1.112.136.153') & (result_df['Proto'] == 'udp')]
    # print(specific_row['mean'].values[0])

    objData = {}
    totalNodes = G.number_of_nodes()
    progress_bar = tqdm(total=totalNodes, desc="Processing", unit="item")
    for node in G.nodes():
        label = 'botnet'
        activityLabel = 1
        if node not in listBotnetAddress:
            label = 'normal'
            activityLabel = 0
        
        # out_data = raw_df[raw_df[['SrcAddr','Proto']].apply(tuple, axis=1).isin([node])]
        # out_srcDesc = out_data['SrcBytes'].describe()
        # in_data = raw_df[raw_df[['DstAddr','Proto']].apply(tuple, axis=1).isin([node])]
        # in_srcDesc = in_data['SrcBytes'].describe()
        out_data = result_src_df.loc[(result_src_df['SrcAddr'] == node[0]) & (result_src_df['Proto'] == node[1])]
        in_data = result_dst_df.loc[(result_dst_df['DstAddr'] == node[0]) & (result_dst_df['Proto'] == node[1])]

        obj={
            'Address' : node[0],
            'Proto': node[1],

            'OutDegree': G.out_degree(node),
            'IntensityOutDegree': G.out_degree(node, weight='weight'),
            'SumSentBytes': 0,
            'MeanSentBytes': 0,
            'MedSentBytes': 0,
            'CVSentBytes': 0,

            'InDegree': G.in_degree(node),
            'IntensityInDegree': G.in_degree(node, weight='weight'),
            'SumReceivedBytes': 0,
            'MeanReceivedBytes': 0,
            'MedReceivedBytes': 0,
            'CVReceivedBytes': 0,

            'Label': label,
            'ActivityLabel': activityLabel
        }

        if(len(out_data) > 0):
            obj['SumSentBytes'] = out_data['sum'].values[0]
            obj['MeanSentBytes'] = out_data['mean'].values[0]
            obj['MedSentBytes'] = out_data['median'].values[0]
            obj['CVSentBytes'] = (out_data['std'].values[0]/out_data['mean'].values[0])*100

        if(len(in_data) > 0):
            obj['SumReceivedBytes'] = in_data['sum'].values[0]
            obj['MeanReceivedBytes'] = in_data['mean'].values[0]
            obj['MedReceivedBytes'] = in_data['median'].values[0]
            obj['CVReceivedBytes'] = (in_data['std'].values[0]/in_data['mean'].values[0])*100
        
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


    


