import uuid
import math

import bin.modules.preProcessing.transform as preProcessing
import bin.modules.utilities.domain as utilities

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *
from bin.helpers.utilities.database import *

timeGapValue = 80
collection = 'sequences'
itemsetCollection = 'itemsets'
sequenceOf = 'SrcAddr' #sequence created base on DstAddr / SrcAddr

def withDataframe(df):
  ctx='Sequential Pattern Miner (With DF)'
  start = watcherStart(ctx)

  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'Unix'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None)
  df['NewLabel'] = df['Label'].apply(preProcessing.labelProcessing)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)

  existIP = ''
  existSegment = ''
  sid = ''
  datasetStartAt = df['Unix'].min()
  datasetEndAt = df['Unix'].max()

  #sequence pattern mining
  for index, row in df.iterrows():
    segment = math.ceil((row['Unix']+1-datasetStartAt)/3600)

    if(existIP == '' or existIP != row[sequenceOf] or row['Diff'] == None
      or row['Diff'] > timeGapValue or segment != existSegment):
      sid = str(uuid.uuid4())
    #in the end of loop
    existIP = row[sequenceOf]
    existSegment = segment
    df.loc[index, 'SequenceId'] = sid
  
  watcherEnd(ctx, start)
  return [], math.ceil((datasetEndAt-datasetStartAt)/3600), df

def withMongo(datasetDetail, df):
  ctx='Sequential Pattern Miner (With Mongo)'
  start = watcherStart(ctx)

  stringDatasetName = datasetDetail['stringDatasetName']
  selected = datasetDetail['selected']

  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  # df['Segment'] = df['Unix'].apply(preProcessing.defineSegment,3).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'Unix'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None)
  df['NewLabel'] = df['Label'].apply(preProcessing.labelProcessing)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)

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

  datasetStartAt = df['Unix'].min()
  datasetEndAt = df['Unix'].max()

  #sequence pattern mining
  for index, row in df.iterrows():
    #set itemset for collection
    # itemset = ( row['DstAddr'], row['NewLabel'] )
    itemset = row['NewLabel']
    segment = math.ceil((row['Unix']+1-datasetStartAt)/3600)
    # Concatenate the values of each row with comma-separated values
    metadata = ','.join([str(val) for val in row.values])

    #######collect itemset for later support counting
    itemsetData = {
      'itemsetId':stringDatasetName+'('+selected+')-'+str(itemset),
      'dataset': stringDatasetName,
      'subDataset': selected,
      'itemset': itemset,
      'support': 0,
      'segment': segment
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
            or segment != existSegment
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
          'segment': segment,
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
          'segment': segment,
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
        'segment': segment,
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
    existSegment = segment
    df.loc[index, 'SequenceId'] = sid

  insertMany(seq, collection)
  insertMany(itemsets, itemsetCollection)
  
  watcherEnd(ctx, start)
  return itemsets, math.ceil((datasetEndAt-datasetStartAt)/3600), df
