import uuid

import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing
import bin.modules.miner.model as model
import bin.modules.machineLearning.domain as ml
import bin.modules.utilities.domain as utilities

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *
from itertools import combinations

timeGapValue = 80
seqWidth = 3600
collection = 'sequences'
itemsetCollection = 'itemsets'
sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

def sequenceMiner(datasetDetail, df):
  ctx='Sequential Pattern Miner'
  start = watcherStart(ctx)

  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df['Segment'] = df['Unix'].apply(preProcessing.defineSegment).fillna(0)
  df = df.sort_values(by=['Segment', sequenceOf, 'Unix'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None)
  df['NewLabel'] = df['Label'].apply(preProcessing.labelProcessing)

  # dictLabel = model.main(model.labelModel, df['NewLabel'].unique())
  # upsertmany((list(model.labelModel.keys())[1]),dictLabel,'Label')

  seq = [] #in one subDataset has one sequence
  itemsets = []
  existIP = ''
  existSegment = ''
  sid = ''
  itemset = ()
  seqTotPkts = 0
  seqTotBytes = 0
  seqSrcBytes = 0
  arrayOfCosine = []

  #temp
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
  # df = df[df['ActivityLabel'] == 'botnet']
  # print(df)
  #sequence pattern mining
  for index, row in df.iterrows():
    #set itemset for collection
    # itemset = ( row['DstAddr'], row['NewLabel'] )
    itemset = row['NewLabel']
    # Concatenate the values of each row with comma-separated values
    metadata = ','.join([str(val) for val in row.values])

    #######collect itemset for later support counting
    itemsetData = {
      'itemsetId':stringDatasetName+'('+selected+')-'+str(itemset),
      'dataset': stringDatasetName,
      'subDataset': selected,
      'itemset': itemset,
      'support': 0,
      'segment': row['Segment']
    }
      #check is data with itemsetId exist in list
    if not any(d.get('itemsetId') == itemsetData['itemsetId'] for d in itemsets):
      itemsets.append(itemsetData)
      #check is data with itemsetId exist in list
    #######collect itemset for later support counting

    if(existIP != '' or existIP == row[sequenceOf]): #same Source IP update Sequence
      if(
        row['Diff'] == None
          or row['Diff'] > timeGapValue
            or row['Segment'] != existSegment
        ):
        #calculate mean cosine while existing sequence finished created
        seq[-1]['meanOfCosineSimilarity'] = utilities.meanOfSimilarity(seq[-1]['arrayOfCosine'])
        seq[-1]['meanSeqTotPkts'] = seq[-1]['seqTotPkts']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqTotBytes'] = seq[-1]['seqTotBytes']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqSrcBytes'] = seq[-1]['seqSrcBytes']/len(seq[-1]['itemset'])
        #calculate mean cosine while existing sequence finished created

        sid = str(uuid.uuid4())
        seqStartTime = row['Unix']
        seqTotPkts = row['TotPkts']
        seqTotBytes = row['TotBytes']
        seqSrcBytes = row['SrcBytes']
        arrayOfCosine = [[seqTotPkts,seqTotBytes,seqSrcBytes]]
        seq.append({
          'sid': sid,
          'segment': row['Segment'],
          'srcAddr': row['SrcAddr'],
          'itemset': [itemset],
          'metadata': [metadata],
          'seqTotPkts': seqTotPkts,
          'seqTotBytes': seqTotBytes,
          'seqSrcBytes': seqSrcBytes,
          'arrayOfCosine': arrayOfCosine,
          'meanSeqTotPkts': 0,
          'meanSeqTotBytes': 0,
          'meanSeqSrcBytes': 0,
          'meanOfCosineSimilarity': 0,
          'datasetSources':stringDatasetName+'('+selected+')',
        })
      elif (row['Diff'] == 0): #while has simultaneous attack
        seqTotPkts += row['TotPkts']
        seqTotBytes += row['TotBytes']
        seqSrcBytes += row['SrcBytes']
        arrayOfCosine.append([seqTotPkts,seqTotBytes,seqSrcBytes])
        seq[-1]['seqTotPkts'] = seqTotPkts
        seq[-1]['seqTotBytes'] = seqTotBytes
        seq[-1]['seqSrcBytes'] = seqSrcBytes
        seq[-1]['itemset'].append(itemset)
        seq[-1]['metadata'].append(metadata)
      else: #same IP in under time gap
        #calculate mean cosine while existing sequence finished created
        seq[-1]['meanOfCosineSimilarity'] = utilities.meanOfSimilarity(seq[-1]['arrayOfCosine'])
        seq[-1]['meanSeqTotPkts'] = seq[-1]['seqTotPkts']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqTotBytes'] = seq[-1]['seqTotBytes']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqSrcBytes'] = seq[-1]['seqSrcBytes']/len(seq[-1]['itemset'])
        #calculate mean cosine while existing sequence finished created
        seqTotPkts = row['TotPkts']
        seqTotBytes = row['TotBytes']
        seqSrcBytes = row['SrcBytes']
        arrayOfCosine = [[seqTotPkts,seqTotBytes,seqSrcBytes]]
        seq.append({
          'sid': sid,
          'segment': row['Segment'],
          'srcAddr': row['SrcAddr'],
          'itemset': [itemset],
          'metadata': [metadata],
          'seqTotPkts': seqTotPkts,
          'seqTotBytes': seqTotBytes,
          'seqSrcBytes': seqSrcBytes,
          'arrayOfCosine' : arrayOfCosine,
          'meanSeqTotPkts': 0,
          'meanSeqTotBytes': 0,
          'meanSeqSrcBytes': 0,
          'meanOfCosineSimilarity': 0,
          'datasetSources':stringDatasetName+'('+selected+')',
        })
    else: #different Source IP make new Sequence, existing sequence finished created
        #calculate mean cosine while existing sequence finished created
      if(len(seq) > 0):
        seq[-1]['meanOfCosineSimilarity'] = utilities.meanOfSimilarity(seq[-1]['arrayOfCosine'])
        seq[-1]['meanSeqTotPkts'] = seq[-1]['seqTotPkts']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqTotBytes'] = seq[-1]['seqTotBytes']/len(seq[-1]['itemset'])
        seq[-1]['meanSeqSrcBytes'] = seq[-1]['seqSrcBytes']/len(seq[-1]['itemset'])
        #calculate mean cosine while existing sequence finished created
      sid = str(uuid.uuid4())
      seqStartTime = row['Unix']
      seqTotPkts = row['TotPkts']
      seqTotBytes = row['TotBytes']
      seqSrcBytes = row['SrcBytes']
      arrayOfCosine = [[seqTotPkts,seqTotBytes,seqSrcBytes]]
      seq.append({
        'sid': sid,
        'segment': row['Segment'],
        'srcAddr': row['SrcAddr'],
        'itemset': [itemset],
        'metadata': [metadata],
        'seqTotPkts': seqTotPkts,
        'seqTotBytes': seqTotBytes,
        'seqSrcBytes': seqSrcBytes,
        'arrayOfCosine' : arrayOfCosine,
        'meanSeqTotPkts': 0,
        'meanSeqTotBytes': 0,
        'meanSeqSrcBytes': 0,
        'meanOfCosineSimilarity': 0,
        'datasetSources':stringDatasetName+'('+selected+')',
      })
    
    #in the end of loop
    existIP = row[sequenceOf]
    existSegment = row['Segment']

  insertMany(seq, collection)
  insertMany(itemsets, itemsetCollection)
  
  watcherEnd(ctx, start)
  return itemsets, len(df['Segment'].unique())

def supportCounter(datasetDetail, itemsets, segment):
  ctx='Support Counter'
  start = watcherStart(ctx)
  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  for element in itemsets:
    query = [
      { '$match' :
        {
          'itemset': element['itemset'],
          'segment' : segment
        }
      },
      {
        '$group' : {
          '_id':'$sid',
          'count': {"$sum":1}
        }
      },
      { '$match' :
        {
          'count': { '$ne': 1}
        }
      },
    ]
    support = aggregate(query,collection)

    updateQuery = {
      'dataset': stringDatasetName,
      'subDataset': selected,
      'itemset': element['itemset'],
      'segment': segment
    }
    modified = {
      '$set': { 'support': len(support) }
    }
    updateOne(updateQuery, modified, itemsetCollection)

  watcherEnd(ctx, start)

def combination(itemsets, segment):
  ctx = 'Itemset Combination'
  start = watcherStart(ctx)
  resultValue = []

  listOfItemsets = [item["itemset"] for item in itemsets if item['segment'] == segment]
  for length in range(3,8):
    for comb in combinations(listOfItemsets,length):
      resultValue.append(list(comb))

  watcherEnd(ctx, start)
  return resultValue

def countCombinationSupport(datasetDetail, itemsets, segment):
  ctx='Combination Support Counter'
  start = watcherStart(ctx)
  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']
  listOfItemset = []

  for element in itemsets:
    query = [
      {
        '$match': {
          'segment' : segment,
          'support' : { '$gt': 2 }
        }
      },
      {
        '$group' : {
          '_id':'$sid',
          'count': {'$sum':1},
          'srcAddr':{"$first":'$srcAddr'},
          'itemset': {'$push':'$itemset'}
        }
      },
      {
          '$addFields':{
              'itemset': {
                  "$reduce": {
                    "input": "$itemset",
                    "initialValue": [],
                    "in": { "$concatArrays": [ "$$value", "$$this" ] }
                  }
              }
          }
      },
      { '$match' :
        {
          'count': { '$ne': 1},
          'itemset': {'$all': element}
        }
      },
    ]
    support = aggregate(query,collection)
    if(len(support) > 0):
      itemsetData = {
        'itemsetId':stringDatasetName+'('+selected+')-'+str(element),
        'srcAddr': support[0]['srcAddr'],
        'dataset': stringDatasetName,
        'subDataset': selected,
        'itemset': str(element),
        'support': len(support)
      }
      listOfItemset.append(itemsetData)

  if(len(listOfItemset) > 0):
    insertMany(listOfItemset, itemsetCollection)
  watcherEnd(ctx, start)

def segmentTraffic(datasetDetail, itemsets, segment):
  ctx='Segement Traffic Analysis'
  start = watcherStart(ctx)
  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']
  listOfItemset = []

  ##get 20 most intense src address
  query = [
    {
      '$match': {
        'segment' : segment,
        'support' : { '$gt': 2 }
      }
    },
    {
      '$group' : {
        '_id':'$sid',
        'count': {'$sum':1},
        'srcAddr':{"$first":'$srcAddr'},
        'itemset': {'$push':'$itemset'}
      }
    },
    {
        '$addFields':{
            'itemset': {
                "$reduce": {
                  "input": "$itemset",
                  "initialValue": [],
                  "in": { "$concatArrays": [ "$$value", "$$this" ] }
                }
            }
        }
    },
    { '$match' :
      {
        'count': { '$ne': 1},
        'itemset': {'$all': element}
      }
    },
  ]
  support = aggregate(query,collection)
  watcherEnd(ctx, start)

def main():
  ctx='Sequential Pattern Mining (Main)'
  start = watcherStart(ctx)
    ##### with input menu
  # datasetName, stringDatasetName, selected = datasetMenu.getData()
    ##### with input menu

    ##### single subDataset
  datasetDetail={
    'datasetName': ncc,
    'stringDatasetName': 'ncc',
    'selected': 'scenario7'
  }
  raw_df = loader.binetflow(
    datasetDetail['datasetName'],
    datasetDetail['selected'],
    datasetDetail['stringDatasetName'])
  df = raw_df.copy() #get a copy from dataset to prevent processed data
  # result = ml.predict(df)
  # raw_df['predictionResult'] = result
  # processed_df = raw_df[raw_df['predictionResult'] == 0] #remove background (ActivityLabel == 1)
  # itemsets, segmentLen = sequenceMiner(datasetDetail, processed_df)
  itemsets, segmentLen = sequenceMiner(datasetDetail, df) #no ML
  # for segment in range(segmentLen):
  #   supportCounter(datasetDetail, itemsets, segment)
  #   combinationItem = combination(itemsets, segment)
  #   countCombinationSupport(datasetDetail, combinationItem, segment)
    ##### single subDataset

  #   ##### loop all dataset
  # for dataset in listAvailableDatasets[:3]:
  #   print(dataset['name'])
  #   for scenario in dataset['list']:
  #     print(scenario)
  #     datasetDetail={
  #       'datasetName': dataset['list'],
  #       'stringDatasetName': dataset['name'],
  #       'selected': scenario
  #     }

  #     df = loader.binetflow(
  #       datasetDetail['datasetName'],
  #       datasetDetail['selected'],
  #       datasetDetail['stringDatasetName'])
  #     df = raw_df.copy() #get a copy from dataset to prevent processed data
  #     result = ml.predict(df)
  #     raw_df['predictionResult'] = result
  #     processed_df = raw_df[raw_df['predictionResult'] == 0] #remove background (ActivityLabel == 1)
  #     itemsets = sequenceMiner(datasetDetail, processed_df)
  #     supportCounter(datasetDetail, itemsets)
  #   ##### loop all dataset

  watcherEnd(ctx, start)

def unique(list1):
  # insert the list to the set
  list_set = set(list1)
  # convert the set to the list
  unique_list = (list(list_set))
  return unique_list
