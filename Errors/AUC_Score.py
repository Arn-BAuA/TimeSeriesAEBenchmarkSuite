import torch
import numpy as np
import copy

class AUCScore:

    def __init__(self,device):
        self.device=device

    def __scoreForClassificationData(self,TrueLabels,PredLabels):
        TP = 0
        FP = 0 
        TN = 0
        FN = 0

        if 1 in TrueLables:
            #By Construction of the DS: all 1, Anomaly
            if 1 in PredLabels:
                TP += 1  
            else:
                FN += 1
        else:
            #all 0, no anomaly
            if 0 in PredLabels:
                TN += 1
            else:
                FP += 1

        return TP,FP,TN,FN

    def __scoreForRegularData(self,TrueLables,PredLabels):
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        acceptanceThreshhold = 0.5

        intervals = []
                          #Start, End, Label
        currentInterval = [0,0,PredLables[0]]
        
        for i in range(1,PredLabels):
            if not PredLabels[i] == currentInterval[2]:
                currentInterval[1] = i-1
                intervals.append(copy(currentInterval))
                currentInterval = [i,i,PredLables[i]]

        currentInterval[1] = len(PredLabels)-1
        intervals.append(currentInterval)
        
        for interval in Intervals:
            labelsRight = TrueLabels[interval[0]:interval[1]] == interval[2]
            right = float(np.count_nonzero(labelsRight))/float(len(labelsRight))

            if right > acceptanceThreshhold:
                #Predicted the right thing... What is that thing?
                if interval[2] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                #Prdiction was wrong
                if interval[2] == 1:
                    FP += 1
                else:
                    FN += 1

        return TP,FP,TN,FN

    def calculate(self,model,DataSet):
        
        if DataSet.IsGeneratedFromClassificationDS:
            score = self.__scoreForClassificationDS
        else:
            score = self.__scoreForRegularData

        Errors = [0]*len(DataSet.Data())
        Labels = DataSet.Labels()
        maxDeviation = 0

        for i in range(0,len(DataSet.Data())):

            seq_true = DataSet.Data()[i].to(self.device)
            seq_pred = model(seq_true)
            
            seq_true = seq_true[0,:,:].to("cpu").detach().numpy()
            seq_pred = seq_pred[0,:,:].to("cpu").detach().numpy()
            
            Errors[i] = np.mean(abs(seq_true-seq_pred),axis = 0)
            if max(Errors[i]) > maxDeviation:
                maxDeviation = max(Errors[i])

        threshholds = np.linspace(0,maxDeviation,100)
        
        TPR=np.zeros(len(threshholds))
        FPR=np.zeros(len(threshholds))

        for i,t in enumerate(threshholds):

            TP = 0 #True Positive
            FP = 0 #False Positive
            TN = 0 #True Negative
            FN = 0 #False Negative

            for j,e in enumerate(Errors):
                predictedAnomalies = e>t #
                resultTP,resultFP,resultTN,resultFN = score(Labels[j],predictedAnomalies)

                TP += resultTP
                FP += resultFP
                TN += resultTN
                FN += resultFN
            
            TPR[i] = float(TP)/float(TP+FN)
            FPR[i] = float(FP)/float(FP+TN)
        

        #AUC According to right square rule quadrature
        rightSQRAUC = 0
        for i in range(1,len(TPR)):
            rightSQRAUC += TPR[i]*(FPR[i]-FPR[i-1])
        #AUC According to left square rule quadrature
        leftSQRAUC = 0
        for i in range(1,len(TPR)):
            leftSQRAUC += TPR[i-1]*(FPR[i]-FPR[i-1])

        #Using the Trapezodoid Rule as AUC measure and
        AUC = (leftSQRAUC + rightSQRAUC)/2
        #the DIfference between the two righthand rules as error
        errAUC = (rightSQRAUC-leftSQRAUC)/2

        return AUC,errAUC

