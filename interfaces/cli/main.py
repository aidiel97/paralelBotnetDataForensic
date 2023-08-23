import time
import os

import helpers.utilities.validator as validator
from helpers.common.globalConfig import PROJECT_NAME

def dumpFunct():
  print('success')

def banner(listMenu):
  x=1
  print("\n\n================================================================")
  print("======| "+PROJECT_NAME+" |=============")
  print("================================================================\n")
  print("Main Menu: ")
  for menu in listMenu:
    print(str(x)+". "+menu[0])
    x += 1
  print("\n================================================================")

def execute(menuIndex, listMenu):
  isValid, menuIndex = validator.validateInput(menuIndex)
  if(isValid and menuIndex < len(listMenu)):
    print("============| Processing Menu: "+str(menuIndex+1)+". "+listMenu[menuIndex][0]+" |========================\n")
    listMenu[menuIndex][1]()

    print("\n\t\t"+listMenu[menuIndex][0]+" Process Success...")
    print("\tback to menu...")
  else:
    print("This Menu Does Not Exist, Please Try Another Input!")
    time.sleep(3) # adding 3 seconds time delay
    os.system("clear")
    menu(listMenu) #call menu again

def menu(listMenu):
  banner(listMenu)
  choose = input("Enter Menu: ")
  execute(choose, listMenu)