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

def generatorWithEdgesArray(G, df, usePkts=False, calculation='sum'): #if usePkts=True, will weighting by total Packet transmitted
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
            objAddress[addressName].append(weight)
        else:
            objAddress[addressName] = [weight]

    for index, row in df.iterrows():
        addressName = row['SrcAddr']+'-'+row['DstAddr']
        if calculation == 'cv':
            array = np.array(objAddress[addressName])
            weight = ( np.std(array)/np.mean(array) )* 100
        elif calculation == 'mean':
            weight = sum(objAddress[addressName]) / len(objAddress[addressName])
        elif calculation == 'median':
            weight = np.median(objAddress[addressName])
        else:
            weight =sum(objAddress[addressName])

        if (row['SrcAddr'],row['DstAddr'],weight) not in edges:
            edges.append((row['SrcAddr'],row['DstAddr'],weight))
            # print((row['SrcAddr'],row['DstAddr'],weight))

    G.add_weighted_edges_from(edges)
