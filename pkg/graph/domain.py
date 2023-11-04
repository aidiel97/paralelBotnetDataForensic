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

from sklearn.preprocessing import StandardScaler 

def graphToTabular(G, raw_df):
    ctx = 'Graph based analysis - Graph to Tabular'
    start = watcherStart(ctx)
    srcId = ['Node-Id']

    # raw_df['ActivityLabel'] = raw_df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
    # botnet = raw_df['ActivityLabel'] == 1
    # botnet_df = raw_df[botnet]
    listBotnetAddress = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    # normal = raw_df['ActivityLabel'] == 0
    # normal_df = raw_df[normal]
    # listNormalAddress = normal_df['SrcAddr'].unique()
    
    #group by sourcebytes
    result_src_df = raw_df.groupby(srcId)['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_dst_df = raw_df.groupby(['DstAddr'])['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_src_df = result_src_df.fillna(0)
    result_dst_df = result_dst_df.fillna(0)

    #group by dur
    dur_src_df = raw_df.groupby(srcId)['Dur'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    dur_dst_df = raw_df.groupby(['DstAddr'])['Dur'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    dur_src_df = dur_src_df.fillna(0)
    dur_dst_df = dur_dst_df.fillna(0)
    
    #group by diff
    diff_src_df = raw_df.groupby(srcId)['Diff'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    diff_dst_df = raw_df.groupby(['DstAddr'])['Diff'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    diff_src_df = diff_src_df.fillna(0)
    diff_dst_df = diff_dst_df.fillna(0)
    
    #group by unix timestamp
    unix_src_df = raw_df.groupby(srcId)['Unix'].agg(['min', 'max']).reset_index()
    unix_dst_df = raw_df.groupby(['DstAddr'])['Unix'].agg(['min', 'max']).reset_index()
    unix_src_df = unix_src_df.fillna(0)
    unix_dst_df = unix_dst_df.fillna(0)

    objData = {}
    totalNodes = G.number_of_nodes()
    progress_bar = tqdm(total=totalNodes, desc="Processing", unit="item")
    for node in G.nodes():
        label = 'botnet'
        activityLabel = 1
        splitNode = node[0].split("-") #node is "147.32.84.181-9" so we need to split "-" to get only the IP
        ipNode = splitNode[0]
        if ipNode not in listBotnetAddress:
            label = 'normal'
            activityLabel = 0
        
        # out_data = raw_df[raw_df[[srcId,'Proto']].apply(tuple, axis=1).isin([node])]
        # out_srcDesc = out_data['SrcBytes'].describe()
        # in_data = raw_df[raw_df[['DstAddr','Proto']].apply(tuple, axis=1).isin([node])]
        # in_srcDesc = in_data['SrcBytes'].describe()
        out_data = result_src_df.loc[result_src_df[srcId[0]] == node[0]]
        in_data = result_dst_df.loc[result_dst_df['DstAddr'] == node[0]]

        dur_out_data = dur_src_df.loc[dur_src_df[srcId[0]] == node[0]]
        dur_in_data = dur_dst_df.loc[dur_dst_df['DstAddr'] == node[0]]
        
        diff_out_data = diff_src_df.loc[diff_src_df[srcId[0]] == node[0]]
        diff_in_data = diff_dst_df.loc[diff_dst_df['DstAddr'] == node[0]]

        unix_out_data = unix_src_df.loc[unix_src_df[srcId[0]] == node[0]]
        unix_in_data = unix_dst_df.loc[unix_dst_df['DstAddr'] == node[0]]

        obj={
            'Address' : node[0],
            # 'Proto': node[1],

            'OutDegree': G.out_degree(node),
            'IntensityOutDegree': G.out_degree(node, weight='weight'),
            'SumSentBytes': 0,
            'MeanSentBytes': 0,
            'MedianSentBytes': 0,
            'StdSentBytes': 0,
            'CVSentBytes': 0,
            'OutSumDur': 0,
            'OutMeanDur': 0,
            'OutMedianDur': 0,
            'OutStdDur': 0,
            'OutSumDiffTime': 0,
            'OutMeanDiffTime': 0,
            'OutMedianDiffTime': 0,
            'OutStdDiffTime': 0,
            'OutStartTime': 0,
            'OutEndTime': 0,

            'InDegree': G.in_degree(node),
            'IntensityInDegree': G.in_degree(node, weight='weight'),
            'SumReceivedBytes': 0,
            'MeanReceivedBytes': 0,
            'MedianReceivedBytes': 0,
            'StdReceivedBytes': 0,
            'CVReceivedBytes': 0,
            'InSumDur': 0,
            'InMeanDur': 0,
            'InMedianDur': 0,
            'InStdDur': 0,
            'InSumDiffTime': 0,
            'InMeanDiffTime': 0,
            'InMedianDiffTime': 0,
            'InStdDiffTime': 0,
            'InStartTime': 0,
            'InEndTime': 0,

            'Label': label,
            'ActivityLabel': activityLabel
        }
        
        #srcBytes
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

        #dur
        if(len(dur_out_data) > 0):
            obj['OutSumDur'] = dur_out_data['sum'].values[0]
            obj['OutMeanDur'] = dur_out_data['mean'].values[0]
            obj['OutMedianDur'] = dur_out_data['median'].values[0]
            obj['OutStdDur'] = dur_out_data['std'].values[0]
            
        if(len(dur_in_data) > 0):
            obj['InSumDur'] = dur_in_data['sum'].values[0]
            obj['InMeanDur'] = dur_in_data['mean'].values[0]
            obj['InMedianDur'] = dur_in_data['median'].values[0]
            obj['InStdDur'] = dur_in_data['std'].values[0]

        #diff
        if(len(diff_out_data) > 0):
            obj['OutSumDiffTime'] = diff_out_data['sum'].values[0]
            obj['OutMeanDiffTime'] = diff_out_data['mean'].values[0]
            obj['OutMedianDiffTime'] = diff_out_data['median'].values[0]
            obj['OutStdDiffTime'] = diff_out_data['std'].values[0]
            
        if(len(diff_in_data) > 0):
            obj['InSumDiffTime'] = diff_in_data['sum'].values[0]
            obj['InMeanDiffTime'] = diff_in_data['mean'].values[0]
            obj['InMedianDiffTime'] = diff_in_data['median'].values[0]
            obj['InStdDiffTime'] = diff_in_data['std'].values[0]
            
        #diff
        if(len(unix_out_data) > 0):
            obj['OutStartTime'] = unix_out_data['min'].values[0]
            obj['OutEndTime'] = unix_out_data['max'].values[0]
            
        if(len(unix_in_data) > 0):
            obj['InStartTime'] = unix_in_data['min'].values[0]
            obj['InEndTime'] = unix_in_data['max'].values[0]
        
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

    raw_df['Unix'] = raw_df['StartTime'].apply(pp.timeToUnix).fillna(0)
    raw_df = raw_df.sort_values(by=['SrcAddr', 'Unix'])
    raw_df = raw_df.reset_index(drop=True)
    
    # Function to calculate the "Diff" column
    def calculate_diff(row):
        index = row.name
        if index > 0 and raw_df.loc[index, 'SrcAddr'] == raw_df.loc[index - 1, 'SrcAddr']:
            return row['Unix'] - raw_df.loc[index - 1, 'Unix']
        return None
    
    raw_df['Diff'] = raw_df.apply(calculate_diff, axis=1)

    # Initialize variables
    x = 0
    prev_src_addr = None
    # Custom function to calculate "Node-Id"
    def calculate_node_id(row):
        nonlocal x, prev_src_addr
        if prev_src_addr is None or row['SrcAddr'] != prev_src_addr:
            x = 0
        elif row['Diff'] > 13:
            x += 1
        prev_src_addr = row['SrcAddr']
        return row['SrcAddr'] + "-" + str(x)

    # Apply the custom function to create the "Node-Id" column
    raw_df['Node-Id'] = raw_df.apply(calculate_node_id, axis=1)
    raw_df['Diff'] = raw_df['Diff'].fillna(0)
    raw_df = raw_df.fillna('-')

    G = nx.DiGraph(directed=True)
    generatorWithEdgesArray(G, raw_df)
    objData = graphToTabular(G, raw_df)
    
    filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
    exportWithArrayOfObject(list(objData.values()), filename)

    watcherEnd(ctx, start)
    return objData


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
