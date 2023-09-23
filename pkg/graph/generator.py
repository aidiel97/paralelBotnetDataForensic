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
    listSrcAddress = df['SrcAddr'].unique()
    listDstAddress = df['DstAddr'].unique()
    
    #start generating graph
    addNodeFromIp(G, listSrcAddress)
    addNodeFromIp(G, listDstAddress)

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        bytes = row['SrcBytes']

        if addressName in objAddress:
            objAddress[addressName].append(bytes)
        else:
            objAddress[addressName] = [bytes]

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        array = np.array(objAddress[addressName])
        
        intensity = len(array)
        cv = ( np.std(array)/np.mean(array) )* 100
        mean = sum(objAddress[addressName]) / len(objAddress[addressName])
        median = np.median(array)
        sumValue = sum(objAddress[addressName])

        weight = (intensity, sumValue, mean, median, cv)

        if (row['SrcAddr'],row['DstAddr'],weight) not in edges:
            edges.append((row['SrcAddr'],row['DstAddr'],weight))

    G.add_weighted_edges_from(edges)
