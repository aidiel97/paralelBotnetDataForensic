import re
import pandas as pd
import socket, struct
import numpy as np

from bin.helpers.utilities.watcher import *
from bin.helpers.common.main import *

def ipToInteger(ip):
  try:
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]
  except OSError:
    return np.nan #return NaN when IP Address is not valid

def labelProcessing(label):
  listOfWord = label.split("-")
  validateArray = []
  for char in listOfWord:
    if(bool(re.search(r'\d', char)) or char == 'From' or char == 'Botnet'):
      continue
    else:
      validateArray.append(char)

  return '-'.join(validateArray)

def normalization(df):
  ctx= '<PRE-PROCESSING> Normalization'
  start = watcherStart(ctx)

  df['Dur'] = ((df['Dur'] - df['Dur'].min()) / (df['Dur'].max() - df['Dur'].min())* 1000000).astype(int)
  df['SrcAddr'] = ((df['SrcAddr'] - df['SrcAddr'].min()) / (df['SrcAddr'].max() - df['SrcAddr'].min())* 1000000).astype(int)  
  df['DstAddr'] = ((df['DstAddr'] - df['DstAddr'].min()) / (df['DstAddr'].max() - df['DstAddr'].min())* 1000000).astype(int)

  watcherEnd(ctx, start)
  return df

#how to use
def transformation(df, ipv4ToInteger=False, oneHotEncode=False):
  ctx= '<PRE-PROCESSING> Transformation'
  start = watcherStart(ctx)
  #make new label for bot prediciton(1/0)
  df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  #transform with dictionary
  df['State']= df['State'].map(stateDict).fillna(0.0).astype(int)
  #transform ip to integer
  df.dropna(subset = ["SrcAddr"], inplace=True)
  df.dropna(subset = ["DstAddr"], inplace=True)

  df['Sport'] = pd.factorize(df.Sport)[0]
  df['Dport'] = pd.factorize(df.Dport)[0]

  if(ipv4ToInteger==True):
    df['SrcAddr'] = df['SrcAddr'].apply(ipToInteger).fillna(0)
    df['DstAddr'] = df['DstAddr'].apply(ipToInteger).fillna(0)

  if(oneHotEncode==True): #transform with  one-hot-encode
    categorical_cols = ['Proto','Dir']
    for col in categorical_cols:
      dummy_cols = pd.get_dummies(df[col], drop_first=True, prefix=col)
      df = pd.concat([df,dummy_cols],axis=1)
      df.drop(columns=col, axis=1, inplace=True)
  else:  #transform with dictionary
    df['Proto']= df['Proto'].map(protoDict).fillna(0.0).astype(int)

  watcherEnd(ctx, start)
  return df
