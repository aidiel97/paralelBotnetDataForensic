"""Paralel Botnet Data Forensic"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import bin.interfaces.cli.main as cli
import bin.modules.spm4Detection.domain as spm4d
import bin.modules.timeAnalysis.domain as timeAnalysis
import bin.modules.miner.domain as miner
import bin.modules.sensorBasedPattern.domain as sbp
import bin.modules.machineLearning.domain as ml

if __name__ == "__main__":
  listMenu = [
    ('Generate Machine Learning Models', ml.modellingWithCTU),
    ('Test Machine Learning Models', ml.executeAllData),
    ('[Single Dataset] Sequential Pattern Mining for Detection', miner.main),
    ('[Test All Dataset] Sequential Pattern Mining for Detection', miner.executeAllData),
    ('Sensor Based Causality Analysis', sbp.main),
    ('Sequence Pattern Mining for Detection', spm4d.main),
    ('Network Time Gap Analysis', timeAnalysis.timeGap),
    ('Network Segment Analysis', timeAnalysis.segmentAnalysis),
  ]
  cli.menu(listMenu)