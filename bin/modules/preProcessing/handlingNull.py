from bin.helpers.utilities.watcher import *

def setEmptyString(df):
  ctx= '<PRE-PROCESSING> Set Empty String'
  start = watcherStart(ctx)

  #need to be confirmed and tested is this the best method
  df['Sport'] = df['Sport'].replace('',0).fillna(0).apply(str).apply(int, base=16)
  df['Dport'] = df['Dport'].replace('',0).fillna(0).apply(str).apply(int, base=16)

  watcherEnd(ctx, start)
  return df