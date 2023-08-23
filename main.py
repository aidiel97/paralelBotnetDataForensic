"""Paralel Botnet Data Forensic"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import interfaces.cli.main as cli
import pkg.spm4Detection.domain as spm4d
import pkg.datasetAnalysis.domain as analysis
import pkg.miner.domain as miner
import pkg.sensorBasedPattern.domain as sbp
import pkg.machineLearning.domain as ml
import pkg.graphVisualization.domain as grp

if __name__ == "__main__":
  listMenu = [
    ('Generate Machine Learning Models', ml.modellingWithCTU),
    ('[Single Dataset] Graph Dataset Generator', grp.singleData),
    ('[Test All Dataset] Graph Dataset Generator', grp.executeAllData),
    ('[Single Dataset]Test Machine Learning Models', ml.singleData),
    ('[Test All Dataset]Test Machine Learning Models', ml.executeAllData),
    ('[Single Dataset] Sequential Pattern Mining for Detection', miner.main),
    ('[Test All Dataset] Sequential Pattern Mining for Detection', miner.executeAllData),
    ('Sensor Based Causality Analysis', sbp.main),
    ('Sequence Pattern Mining for Detection', spm4d.main),
    ('Network Time Gap Analysis', analysis.timeGap),
    ('Network Traffic Packet Source Bytes Analysis', analysis.SrcBytes),
    ('Sequence Analysis', analysis.sequence),
    ('Segment Analysis', analysis.segmentAnalysis),
    ('Export Binetflow Dataset', analysis.exportDataset),
    ('Export All Categorical Feature Unique', analysis.exportAllCategoricalFetureUnique),
    ('Network Actor Analysis', analysis.networkActor),
  ]
  cli.menu(listMenu)