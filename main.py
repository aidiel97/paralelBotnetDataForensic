"""Dimentional Reduction on Botnet Data"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import bin.interfaces.cli.main as cli
from bin.modules.domain import *

if __name__ == "__main__":
  listMenu = [
    ('Sensor Based Causality Analysis', sensorBasedCausalityAnalysis)
  ]
  cli.menu(listMenu)