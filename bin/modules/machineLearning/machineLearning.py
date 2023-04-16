from bin.helpers.utilities.watcher import *
import bin.helpers.utilities.csvGenerator as csv

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


algorithmDict = {
  'decisionTree': DecisionTreeClassifier(),
  'randomForest': RandomForestClassifier(n_estimators=100),
  'svc' : SVC(kernel='linear', C=10.0, random_state=1),
  'naiveBayes': GaussianNB(),
  'knn': KNeighborsClassifier(),
  'logisticRegression' : LogisticRegression(C=10000, solver='liblinear')
}

def modelFileName(algorithm): return 'collections/'+algorithm+'.pkl'

def modelling(x, y, algorithm='randomForest'):
  ctx= 'Training Model With: '+algorithm.upper()
  start= watcherStart(ctx)

  model = algorithmDict[algorithm]
  model.fit(x, y)
  pickle.dump(model, open(modelFileName(algorithm), 'wb'))

  watcherEnd(ctx, start)

def classification(x, algorithm='randomForest'):
  ctx= 'Classifying Data'
  start= watcherStart(ctx)

  model = pickle.load(open(modelFileName(algorithm), 'rb'))
  predictionResult = model.predict(x)

  watcherEnd(ctx, start)
  return predictionResult

def evaluation(ctx, y, predictionResult, algorithm='randomForest'):
  tn, fp, fn, tp = confusion_matrix(y, predictionResult).ravel()

  print('\nTotal input data\t\t\t: '+str(y.shape[0]))
  print('TN (predict result 0, actual 0)\t\t: '+str(tn))
  print('FP (predict result 1, actual 0)\t\t: '+str(fp))
  print('FN (predict result 0, actual 1)\t\t: '+str(fn))
  print('TP (predict result 1, actual 1)\t\t: '+str(tp))
  print('Accuracy\t\t\t\t: '+str((tp+tn)/(tp+tn+fp+fn)))

  csv.classificationResult(ctx, algorithm, tn, fp, fn, tp)