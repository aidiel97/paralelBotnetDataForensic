import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as pp
import pkg.machineLearning.machineLearning as ml

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.database import *
from helpers.utilities.csvGenerator import exportWithArrayOfObject

def addNodeFromIp(G, arrayOfIp):
    for element in arrayOfIp:
        G.add_node(element)

def generator(G, df):
    objAddress = {}
    listSrcAddress = df['SrcAddr'].unique()
    listDstAddress = df['DstAddr'].unique()

    #start generating graph
    addNodeFromIp(G, listSrcAddress)
    addNodeFromIp(G, listDstAddress)

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        if addressName in objAddress:
            objAddress[addressName] += 1
        else:
            objAddress[addressName] = 1

        weight =objAddress[addressName]
        G.add_edge(row['SrcAddr'],row['DstAddr'], weight=weight)

def exportGraph(G, filename):
    pos = nx.planar_layout(G, scale=2)
    # pos = nx.spring_layout(G, scale=2)
    # pos = nx.spiral_layout(G, scale=2)
    # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=1000)

    # node labels
    # nx.draw_networkx_labels(G, pos, font_size=8, font_family="Times New Roman")

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=8, font_family="Times New Roman", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_family="Times New Roman", font_color='red')

    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def generatorWithEdgesArray(G, df, usePkts=False): #if usePkts=True, will weighting by total Packet transmitted
    edges = []
    objAddress = {}
    listSrcAddress = df['SrcAddr'].unique()
    listDstAddress = df['DstAddr'].unique()
    
    #start generating graph
    addNodeFromIp(G, listSrcAddress)
    addNodeFromIp(G, listDstAddress)

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        if usePkts:
            weight = row['SrcBytes']
        else:
            weight = 1

        if addressName in objAddress:
            objAddress[addressName] += weight
        else:
            objAddress[addressName] = weight

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        weight =objAddress[addressName]
        if (row['SrcAddr'],row['DstAddr'],weight) not in edges:
            edges.append((row['SrcAddr'],row['DstAddr'],weight))
            print((row['SrcAddr'],row['DstAddr'],weight))

    G.add_weighted_edges_from(edges)

def singleData():
    ctx = 'Graph based analysis - Single Dataset'
    start = watcherStart(ctx)
    ##### single subDataset
    datasetDetail = {
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

    datasetInSpecificBotnet = ['147.32.84.165','147.32.84.191','147.32.84.192']
    raw_df['Label'] = raw_df['Label'].apply(pp.labelSimplier)
    botnet = raw_df[raw_df['Label'] == 'botnet']
    normal = raw_df.loc[(raw_df['Label'] == 'normal') & (raw_df['DstAddr'].isin(datasetInSpecificBotnet))]
    others = raw_df[raw_df['DstAddr'].isin(datasetInSpecificBotnet)]
    # withoutBackground = raw_df[raw_df['Label'] != 'background']
    listBotnetAddress = botnet['SrcAddr'].unique()
    listNormalAddress = normal['SrcAddr'].unique()

    G = nx.DiGraph(directed=True)
    generatorWithEdgesArray(G, botnet)
    exportGraph(G, 'collections/graph.png')

    # normG = nx.DiGraph(directed=True)
    # generatorWithEdgesArray(normG, others)
    # exportGraph(normG, 'collections/norm-graph.png')
    
    # NG = nx.DiGraph(directed=True)
    # generatorWithEdgesArray(NG, botnet, True)
    # exportGraph(NG, 'collections/weighted-graph.png')

    # objData = {}
    # for node in G.nodes():
    #     label = 2 #need to try multilable detection
    #     if node in listBotnetAddress:
    #         label = 1
    #     if node in listNormalAddress:
    #         label = 2
    #     obj = {
    #         'Address': node,
    #         'WeightedOutDegree': G.out_degree(node, weight='weight'),
    #         'WeightedInDegree': G.in_degree(node, weight='weight'),
    #         'OutDegree': G.out_degree(node),
    #         'InDegree': G.in_degree(node),
    #         'TotSentBytes': NG.out_degree(node, weight='weight'),
    #         'TotReceivedBytes': NG.in_degree(node, weight='weight'),
    #         'Label': label
    #     }
    #     objData[node] = obj
    
    # filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
    # exportWithArrayOfObject(list(objData.values()), filename)

    # df = pd.DataFrame(objData.values())
    
    # x = df.drop(['Label', 'Address'],axis=1)
    # y = df['Label']
    
    # ml.modelling(x, y, 'randomForest')
    # predictionResult = ml.classification(x, 'randomForest')
    # print(predictionResult)
    # ml.evaluation(ctx, y, predictionResult, 'randomForest')

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

        raw_df = loader.binetflow(
            datasetDetail['datasetName'],
            datasetDetail['selected'],
            datasetDetail['stringDatasetName'])

        raw_df['Label'] = raw_df['Label'].apply(pp.labelSimplier)
        botnet = raw_df[raw_df['Label'] == 'botnet']
        listBotnetAddress = botnet['SrcAddr'].unique()
        
        G = nx.DiGraph()
        generatorWithEdgesArray(G, raw_df)
        
        NG = nx.DiGraph()
        generatorWithEdgesArray(NG, raw_df, True)

        objData = {}
        for node in G.nodes():
            label = 'botnet'
            if node not in listBotnetAddress:
                label = 'normal'
            obj = {
                'Address': node,
                'WeightedOutDegree': G.out_degree(node, weight='weight'),
                'WeightedInDegree': G.in_degree(node, weight='weight'),
                'OutDegree': G.out_degree(node),
                'InDegree': G.in_degree(node),
                'TotSentBytes': NG.out_degree(node, weight='weight'),
                'TotReceivedBytes': NG.in_degree(node, weight='weight'),
                'Label': label
            }
            objData[node] = obj
        
        filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
        # exportWithArrayOfObject(list(objData.values()), filename)

        df = pd.DataFrame(objData.values())
        
        x = df.drop(['Label', 'Address'],axis=1)
        y = df['Label']
        
        ml.modelling(x, y, 'randomForest')
        predictionResult = ml.classification(x, 'randomForest')
        print(predictionResult)
        ml.evaluation(ctx, y, predictionResult, 'randomForest')

  ##### loop all dataset

  watcherEnd(ctx, start)


    


