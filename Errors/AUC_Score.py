import torch
import numpy as np

class AUCScore:

    def __init__(self,device):
        self.device=device

    def __scoreForClassificationData(self,TrueLabels,FalseLabels):
        pass

    def __scoreForRegularData(self,TrueLables,FalseLabels):
        pass

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

