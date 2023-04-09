import uuid

import bin.helpers.utilities.dataLoader as loader
import bin.modules.preProcessing.transform as preProcessing
import bin.modules.miner.model as model

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *

timeGapValue = 80
collection = 'sequences'
itemsetCollection = 'itemsets'
sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

def sequenceMiner(datasetDetail, df):
  ctx='Sequential Pattern Miner'
  start = watcherStart(ctx)

  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'StartTime'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None)
  df['NewLabel'] = df['Label'].apply(preProcessing.labelProcessing)

  dictLabel = model.main(model.labelModel, df['NewLabel'].unique())
  upsertmany((list(model.labelModel.keys())[1]),dictLabel,'Label')

  seq = [] #in one subDataset has one sequence
  itemsets = []
  existIP = ''
  sid = ''
  itemset = ()

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

  #collect itemset for later support counting
    itemsetData = {
      'itemsetId':stringDatasetName+'('+selected+')-'+str(itemset),
      'dataset': stringDatasetName,
      'subDataset': selected,
      'itemset': itemset,
      'support': 0,
      'metadata': []
    }
      #check is data with itemsetId exist in list
    if not any(d.get('itemsetId') == itemsetData['itemsetId'] for d in itemsets):
      itemsets.append(itemsetData)
      #check is data with itemsetId exist in list
  #collect itemset for later support counting

    if(existIP != '' or existIP == row[sequenceOf]):
      if(row['Diff'] == None or row['Diff'] > timeGapValue):
        sid = str(uuid.uuid4())
        seq.append({
          'sid': sid,
          'srcAddr': row['SrcAddr'],
          'itemset': [itemset],
          'metadata': [metadata]
        })
      elif (row['Diff'] == 0): #while has simultaneous attack
        seq[-1]['itemset'].append(itemset)
        seq[-1]['metadata'].append(metadata)
      else:
        seq.append({
          'sid': sid,
          'srcAddr': row['SrcAddr'],
          'itemset': [itemset],
          'metadata': [metadata]
        })
    else:
      existIP = row[sequenceOf]
      sid = str(uuid.uuid4())
      seq.append({
        'sid': sid,
        'srcAddr': row['SrcAddr'],
        'itemset': [itemset],
        'metadata': [metadata]
      })

  insertMany(seq, collection)
  insertMany(itemsets, itemsetCollection)
  
  watcherEnd(ctx, start)
  return itemsets

def supportCounter(datasetDetail, itemsets):
  ctx='Support Counter'
  start = watcherStart(ctx)
  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  for element in itemsets:
    query = [
      { '$match' :
        {
          'itemset': element['itemset']
        }
      },
      {
        '$group' : {
          '_id':'$sid',
          'count': {"$sum":1}
        }
      }
    ]
    support = aggregate(query,collection)

    updateQuery = {
      'dataset': stringDatasetName,
      'subDataset': selected,
      'itemset': element['itemset']
    }
    modified = {
      '$set': {'support': len(support), 'metadata': support}
    }
    updateOne(updateQuery,modified, itemsetCollection)

  watcherEnd(ctx, start)

def main():
  ctx='Sequential Pattern Mining (Main)'
  start = watcherStart(ctx)

  # datasetName, stringDatasetName, selected = datasetMenu.getData()

  # datasetName = ctu
  # stringDatasetName = 'ctu'
  # selected = 'scenario7'
  # df = loader.binetflow(datasetName, selected, stringDatasetName)

  for dataset in listAvailableDatasets[:3]:
    print(dataset['name'])
    for scenario in dataset['list']:
      print(scenario)
      datasetName = dataset['list']
      stringDatasetName = dataset['name']
      selected = scenario

      df = loader.binetflow(datasetName, selected, stringDatasetName)

      datasetDetail={
        'datasetName': datasetName,
        'stringDatasetName': stringDatasetName,
        'selected': selected
      }
      itemsets = sequenceMiner(datasetDetail, df)
      supportCounter(datasetDetail, itemsets)

  watcherEnd(ctx, start)

def unique(list1):
  # insert the list to the set
  list_set = set(list1)
  # convert the set to the list
  unique_list = (list(list_set))
  return unique_list
