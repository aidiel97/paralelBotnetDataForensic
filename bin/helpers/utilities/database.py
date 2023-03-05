import os
import bson
import pymongo

from bin.helpers.utilities.watcher import *
from bin.helpers.common.globalConfig import MONGO_URL, MONGO_DATABASE, MONGO_COLLECTION_DEFAULT, MONGO_DUMP_PATH

mongoClient = pymongo.MongoClient(MONGO_URL) #better user .env
database = mongoClient[MONGO_DATABASE] #better user .env
defaultCollection = MONGO_COLLECTION_DEFAULT #better user .env

dumpPath = MONGO_DUMP_PATH
dumpDefaultCollections = ['sequentialActivities', 'uniquePattern']

#commmand
def insertOne(dictData, collectionName=defaultCollection):
  collection = database[collectionName]
  row = collection.insert_one(dictData)
  # print("success insert with id: "+str(row.inserted_id))

def insertMany(listData, collectionName=defaultCollection):
  collection = database[collectionName]
  row = collection.insert_many(listData)
  # print("\t..Success insert into "+collectionName)

def upsertOne(query, record, collectionName=defaultCollection):
  collection = database[collectionName]
  collection.replace_one(query, record, upsert=True)
  # print("Success upsert ", str(query))

def updateMany(query, record, collectionName=defaultCollection):
  collection = database[collectionName]
  collection.update_many(query, record)
  # print("\t..Success updateMany")

def deleteOne(query, collectionName=defaultCollection):
  collection = database[collectionName]
  collection.delete_one(query)
  print("\t..Success delete one", str(query))

def deleteMany(query, collectionName=defaultCollection):
  collection = database[collectionName]
  collection.delete_many(query)
  print("\t..Success delete many ", str(query))

#query
def findOne(query={}, collectionName=defaultCollection):
  collection = database[collectionName]
  doc = collection.find_one(query)

  return doc

def aggregate(pipeline, collectionName=defaultCollection):
  result = []
  collection = database[collectionName]
  doc = collection.aggregate(pipeline)

  for record in doc:
    result.append(record)

  return result

def dump(collections=dumpDefaultCollections):
  ctx='MONGO-DUMP'
  start = watcherStart(ctx)
  print('\n')
  for coll in collections:
    with open(os.path.join(dumpPath, f'{coll}.bson'), 'wb+') as f:
      for doc in database[coll].find():
        f.write(bson.BSON.encode(doc))
    
    print('........success dump into '+coll+'.bson')
  
  watcherEnd(ctx, start)

def restore():
  ctx='MONGO-RESTORE'
  start = watcherStart(ctx)
  print('\n')
  for coll in os.listdir(dumpPath):
    if coll.endswith('.bson'):
      with open(os.path.join(dumpPath, coll), 'rb+') as f:
        database[coll.split('.')[0]].insert_many(bson.decode_all(f.read()))
              
    print('........success restore '+coll+' into mongoDB')
  
  watcherEnd(ctx, start)