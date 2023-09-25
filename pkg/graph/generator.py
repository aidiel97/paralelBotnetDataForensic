import numpy as np

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
    edges = []
    objAddress = {}
    
    columns_of_interest = ['SrcAddr', 'Proto', 'Sport']
    subset_df = df[columns_of_interest]
    unique_combinations = subset_df.drop_duplicates()
    listSrcAddress = [tuple(x) for x in unique_combinations.values]

    dst_columns_of_interest = ['DstAddr', 'Proto', 'Dport']
    dst_subset_df = df[dst_columns_of_interest]
    dst_unique_combinations = dst_subset_df.drop_duplicates()
    listDstAddress = [tuple(x) for x in dst_unique_combinations.values]
    
    #start generating graph
    addNodeFromIp(G, listSrcAddress)
    addNodeFromIp(G, listDstAddress)

    for index, row in df.iterrows():
        addressName = str((row['SrcAddr'],row['Proto'],row['Sport']))+'-'+str((row['DstAddr'],row['Proto'],row['Dport']))
        bytes = row['SrcBytes']

        if addressName in objAddress:
            objAddress[addressName].append(bytes)
        else:
            objAddress[addressName] = [bytes]

    for index, row in df.iterrows():
        addressName = str((row['SrcAddr'],row['Proto'],row['Sport']))+'-'+str((row['DstAddr'],row['Proto'],row['Dport']))
        array = np.array(objAddress[addressName])
        
        intensity = len(array)
        cv = ( np.std(array)/np.mean(array) )* 100
        mean = sum(objAddress[addressName]) / len(objAddress[addressName])
        median = np.median(array)
        sumValue = sum(objAddress[addressName])

        weight = (intensity, sumValue, mean, median, cv)
        source = (row['SrcAddr'],row['Proto'],row['Sport'])
        dest = (row['DstAddr'],row['Proto'],row['Dport'])

        if (source,dest,weight) not in edges:
            edges.append((source,dest,weight))

    G.add_weighted_edges_from(edges)
