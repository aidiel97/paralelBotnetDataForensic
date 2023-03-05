import csv
from datetime import datetime
from bin.helpers.utilities.watcher import *

def similarityResult(botnetUniqueSelection, normalUniqueSelection, botnetRecord, normalRecord, samplingRule, decompositor, similarityResult):
  ctx= 'Export Similarity Result'
  start= watcherStart(ctx)
  # list of column names 
  field_names = ['CreatedAt', 'botnetUniqueSelection','normalUniqueSelection','botnetRecord','normalRecord','samplingRule', 'decompositor', 'similarityResult']
  # Dictionary
  dict = {
    "CreatedAt": datetime.now(),
    "botnetUniqueSelection": botnetUniqueSelection,
    "normalUniqueSelection": normalUniqueSelection,
    "botnetRecord": botnetRecord,
    "normalRecord": normalRecord,
    "samplingRule": samplingRule,
    "decompositor": decompositor,
    "similarityResult": similarityResult
  }

  with open('data/similarity_results.csv', 'a', newline='') as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
    dict_object.writerow(dict)

  watcherEnd(ctx, start)

def classificationResult(classificationContext, algorithm, tn, fp, fn, tp):
  ctx= 'Export Classification Result'
  start= watcherStart(ctx)
  # list of column names 
  field_names = ['CreatedAt','Algorithm','TN','FP','FN', 'TP', 'ClassificationContext']
  # Dictionary
  dict = {
    "CreatedAt": datetime.now(),
    "Algorithm":algorithm,
    "ClassificationContext":classificationContext,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
  }

  with open('collections/classification_results.csv', 'a', newline='') as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
    dict_object.writerow(dict)

  watcherEnd(ctx, start)