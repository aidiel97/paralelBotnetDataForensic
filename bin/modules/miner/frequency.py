from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *
from itertools import combinations

collection = 'sequences'
itemsetCollection = 'itemsets'

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

def segmentTraffic(datasetDetail, segment):
  ctx='Segement Traffic Analysis'
  start = watcherStart(ctx)
  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  ##get 20 most intense src address
  query = [
    {
      '$match': {
        'datasetSources': stringDatasetName+'('+selected+')',
        'segment' : segment
      }
    },
    {
        '$group' : {
          '_id':'$srcAddr',
          'srcAddr':{"$first":'$srcAddr'},
          'SeqSrcBytes': {"$sum":"$meanSeqSrcBytes"},
          'SeqTotBytes': {"$sum":"$meanSeqTotBytes"},
          'SeqTotPkts': {"$sum":"$meanSeqTotPkts"}
        }
      },
      {
          '$sort':{
              'SeqSrcBytes': -1,
              'SeqTotPkts': 1,
              'SeqTotBytes': -1,
          }
      }
  ]
  result = aggregate(query,collection)

  watcherEnd(ctx, start)
  return result
