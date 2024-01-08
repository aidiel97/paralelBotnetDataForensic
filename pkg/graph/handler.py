import networkx as nx
from tqdm import tqdm
import time

import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as pp

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.database import *
from helpers.utilities.csvGenerator import exportWithArrayOfObject
from pkg.graph.models import *
from pkg.graph.generator import *
from pkg.graph.extractor import *

def dftoGraph(datasetDetail):
    ctx = 'Graph based analysis - DF to Graph'
    start = watcherStart(ctx)

    # check the varaible is string or dictionary
    if isinstance(datasetDetail, str):
        raw_df = loader.rawCsv(datasetDetail)
    else:
        raw_df = loader.binetflow(
            datasetDetail['datasetName'],
            datasetDetail['selected'],
            datasetDetail['stringDatasetName'])
    
    raw_df['Unix'] = raw_df['StartTime'].apply(pp.timeToUnix).fillna(0)
    
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
    # Custom function to calculate "Src-Id"
    def calculate_src_id(row):
        nonlocal x, prev_src_addr
        if prev_src_addr is None or row['SrcAddr'] != prev_src_addr:
            x = 0
        elif row['Diff'] > DEFAULT_TIME_GAP:
            x += 1
        prev_src_addr = row['SrcAddr']
        return row['SrcAddr'] + "-" + str(x)

    # Custom function to calculate "Src-Id"
    prev_dst_addr = None
    prev_srcId_addr = None
    def calculate_dst_id(row):
        nonlocal x, prev_dst_addr, prev_srcId_addr
        if (prev_dst_addr is None and prev_dst_addr is None) or row['DstAddr'] != prev_dst_addr:
            x = 0
        elif row['Src-Id'] != prev_srcId_addr:
            x += 1
        prev_dst_addr = row['DstAddr']
        prev_srcId_addr = row['Src-Id']
        return row['DstAddr'] + "-" + str(x)

    raw_df = raw_df.sort_values(by=['SrcAddr', 'Unix'])
    raw_df = raw_df.reset_index(drop=True)
    raw_df['Src-Id'] = raw_df.apply(calculate_src_id, axis=1)

    raw_df = raw_df.sort_values(by=['DstAddr', 'Src-Id'])
    raw_df = raw_df.reset_index(drop=True)
    raw_df['Dst-Id'] = raw_df.apply(calculate_dst_id, axis=1)

    raw_df['Diff'] = raw_df['Diff'].fillna(0)
    raw_df = raw_df.fillna('-')

    extractGraph(raw_df, datasetDetail)

    # G = nx.DiGraph(directed=True)
    # generatorWithEdgesArray(G, raw_df)
    # objData = graphToTabular(G, raw_df)
    
    # filename = 'collections/'+datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']+'.csv'
    # exportWithArrayOfObject(list(objData.values()), filename)

    watcherEnd(ctx, start)

def graphToTabular(G, raw_df):
    ctx = 'Graph based analysis - Graph to Tabular'
    start = watcherStart(ctx)
    srcId = ['Node-Id']

    listBotnetAddress = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

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
