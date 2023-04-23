import time

def watcherStart(processName):
  start = time.time()
  print('\n===========================================| [ START ] '+processName)

  return start

def watcherEnd(processName, start=time.time()):
  end = time.time()
  processingTime = '{:.3f}'.format(end - start)
  print('\n===========================================| [  END  ] '+processName+' ('+str(processingTime)+' s)')
