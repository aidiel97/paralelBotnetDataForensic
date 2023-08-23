from helpers.utilities.watcher import watcherStart, watcherEnd

def featureDropping(df, features):
  ctx= 'Data Feature Dropping'
  start = watcherStart(ctx)
  for feature in features:
    df.drop(columns=feature, inplace=True)

  watcherEnd(ctx, start)
  return df
