import numpy as np
from helpers.utilities.watcher import *

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

def generatorWithEdgesArray(G, df): #if usePkts=True, will weighting by total Packet transmitted
    ctx = 'Graph based analysis - Generator'
    start = watcherStart(ctx)
    edges = []
    objAddress = {}
    
    columns_of_interest = ['Node-Id']
    subset_df = df[columns_of_interest]
    unique_combinations = subset_df.drop_duplicates()
    listSrcAddress = [tuple(x) for x in unique_combinations.values]

    dst_columns_of_interest = ['DstAddr']
    dst_subset_df = df[dst_columns_of_interest]
    dst_unique_combinations = dst_subset_df.drop_duplicates()
    listDstAddress = [tuple(x) for x in dst_unique_combinations.values]
    
    #start generating graph
    addNodeFromIp(G, listSrcAddress)
    addNodeFromIp(G, listDstAddress)
    for index, row in df.iterrows():
        addressName = row['Node-Id']+'-'+row['DstAddr']

        if addressName in objAddress:
            objAddress[addressName] += 1
        else:
            objAddress[addressName] = 1

    for index, row in df.iterrows():
        addressName = row['Node-Id']+'-'+row['DstAddr']
        
        weight = objAddress[addressName]
        source = row['Node-Id']
        dest = row['DstAddr']

        if (source,dest,weight) not in edges:
            edges.append((source,dest,weight))

    G.add_weighted_edges_from(edges)
    
    watcherEnd(ctx, start)
