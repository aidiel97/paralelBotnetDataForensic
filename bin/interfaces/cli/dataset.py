import os
import time
import bin.helpers.utilities.validator as validator
from bin.helpers.common.main import listAvailableDatasets
import bin.interfaces.cli.main as main

def getListDatasetMenu():
  os.system("clear")
  x=1
  print("\n\n------------------------------------------")
  print("\t| List Available Datasets |")
  print("------------------------------------------\n")
  for dataset in listAvailableDatasets:
    print(str(x)+". "+dataset['name'])
    x += 1
  print("\n------------------------------------------")
  choose = input("Choose one dataset: ")
  isValid, choose = validator.validateInput(choose)
  if(isValid == False or choose >= len(listAvailableDatasets)):
    print("This Option Does Not Exist, Bring You Back to Main Menu!")
    time.sleep(3) # adding 3 seconds time delay
    os.system("clear")
    main.menu() #call menu again
  return choose

def getListDatasetDetailMenu(datasetIndex):
  x=1
  os.system("clear")
  chooseDataset = listAvailableDatasets[datasetIndex]
  listSubDataset = list(chooseDataset['list'].keys())
  
  print("\n\n------------------------------------------")
  print("\t| List Sub Dataset on "+chooseDataset['name']+" |")
  print("------------------------------------------\n")
  
  for subDataset in listSubDataset:
    print(str(x)+". "+subDataset)
    x += 1
  
  print("\n------------------------------------------")
  choose = input("Choose one subDataset: ")
  isValid, choose = validator.validateInput(choose)
  if(isValid == False or choose >= len(listSubDataset)):
    print("This Option Does Not Exist, Bring You Back to Main Menu!")
    time.sleep(3) # adding 3 seconds time delay
    os.system("clear")
    main.menu() #call menu again
  return choose+1

def getData():
  datasetIndex = getListDatasetMenu()
  datasetName = listAvailableDatasets[int(datasetIndex)]['list']
  datasetDetail = getListDatasetDetailMenu(datasetIndex)
  stringDatasetName = listAvailableDatasets[int(datasetIndex)]['shortName']
  selected = 'scenario'+str(datasetDetail)

  return datasetName, stringDatasetName, selected