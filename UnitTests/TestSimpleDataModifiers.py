##############################################
# Unit test method to test the simpler datamodifiers.

import unittest

import numpy as np
import torch
from BlockAndDatablock import DataBlock
from DataModifyers.resample import resample

class TestResample(unittest.TestCase):

    def test_linearFunction(self):
        nBegins = [10,50,50]
        nInterps = [100,100,10]
        
        
        for i in range(0,len(nBegins)):
            
            nBegin = nBegins[i]
            nInterp = nInterps[i]

            StartData = torch.Tensor([np.linspace(0,2,nBegin)])
            StartLabels = np.zeros(nBegin)
            StartLabels[0:int(0.25*nBegin)] = 1
            
            IntData = torch.Tensor([np.linspace(0,2,nInterp)])
            IntLabels = np.zeros(nInterp)
            nOnesOriginal = np.sum(StartLabels == 1)
            ratio = (nOnesOriginal-0.5)/(nBegin-1)
            IntLabels[0:int(ratio*nInterp)] = 1
        
            DataSet = DataBlock("TestBlock",[StartData],1)
            DataSet.setLabels([StartLabels])

            InterpolatedSet = resample(nInterp,DataSet)
            outData = InterpolatedSet.Data()[0]
            outLabels = InterpolatedSet.Labels()[0]


            self.assertEqual(torch.equal(torch.round(outData,decimals = 4),torch.round(IntData,decimals = 4)),True)
            self.assertEqual(abs(np.sum(outLabels)-np.sum(IntLabels))<= 1,True)
        

if __name__ == "__main__":
    unittest.main()
