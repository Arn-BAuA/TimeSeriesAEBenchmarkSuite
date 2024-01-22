##############################################
# Unit test method to test the simpler datamodifiers.

import unittest

import numpy as np
import torch
from BlockAndDatablock import DataBlock
from DataModifyers.resample import resample

class TestResample(unittest.TestCase):

    def test_linearFunction(self):
        nBegin = 10
        nInterp = 20

        StartData = torch.Tensor(np.linspace(0,2,nBegin))
        StartLabels = np.zeros(nBegin)
        StartLabels[0:int(0.25*nBegin)] = 1
        
        IntData = torch.Tensor(np.linspace(0,2,nInterp))
        IntLabels = np.zeros(nBegin)
        IntLabels[0:int(0.25*nInterp)] = 1
    
        DataSet = DataBlock("TestBlock",[StartData],1)
        DataSet.setLabels(StartLabels)

        InterpolatedSet = resample(nInterp,DataSet)
        outData = InterpolatedSet.Data[0]
        outLabels = InterpolatedSet.Labels[0]

        self.assertEqual(outData,IntData)
        self.assertEqual(outLabels,IntLabels)


if __name__ == "__main__":
    unittest.main()
