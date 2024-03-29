import torch
try:
    #Using cupy turns out to sometimes make things worse due to excessive coputation cost for creating large arrays.
    jalt

    import cupy as np
    usingCupy = True
    print("AUC Score: Using cupy")
except:
    import numpy as np
    usingCupy = False
    print("AUC Score: Using nupy")

from copy import copy


#We have two internal scores processing the
#predictions of the algorithm. This is, because 
#the labeling is not consistent in the framework.
#see DataBlock for explaination.
def scoreForClassificationData(TrueLabels,error,treshhold):
    #here,Error is a number

    PredLabel = error>treshhold

    TP = 0 #True Positive
    FP = 0 #False Positive
    TN = 0 #True Negative
    FN = 0 #False Negative

    if 1 in TrueLabels:
        #By Construction of the DS: all 1, Anomaly
        if PredLabel:
            TP += 1 
        else:
            FN += 1
    else:
        #all 0, no anomaly
        if not PredLabel:
            TN += 1
        else:
            FP += 1

    return TP,FP,TN,FN

def scorePointwise(TrueLabels,error,treshhold):

    PredLabels = error>treshhold 

    TP = np.count_nonzero(np.logical_and(np.equal(PredLabels,TrueLabels),np.equal(PredLabels, True)))
    FP = np.count_nonzero(np.logical_and(np.equal(PredLabels,np.logical_not(TrueLabels)),np.equal(PredLabels,True)))
    TN = np.count_nonzero(np.logical_and(np.equal(PredLabels,TrueLabels),np.equal(PredLabels, False)))
    FN = np.count_nonzero(np.logical_and(np.equal(PredLabels,np.logical_not(TrueLabels)),np.equal(PredLabels,False)))
    
    return TP,FP,TN,FN

# Not Practical at the moment:
# The problem here is, that the intervals are countet as 1 (TP or FP or TN or FN). So the absolute number of TP, FP, ... 
# depents on the number of intervals, which depents on the threshhold value. This can lead to a situation, where the
# Monotony for the ROC curve is not given.
# This needs to be fixed.

def scoreForRegularData(TrueLabels,error,treshhold):
    
    PredLabels = error>treshhold 
    
    TP = 0 #True Positive
    FP = 0 #False Positive
    TN = 0 #True Negative
    FN = 0 #False Negative

    acceptanceThreshhold = 0.5

    intervals = []
                      #Start, End, Label
    currentInterval = [0,0,PredLabels[0]]
    
    for i in range(1,len(PredLabels)):
        if not PredLabels[i] == currentInterval[2]:
            currentInterval[1] = i-1
            intervals.append(copy(currentInterval))
            currentInterval = [i,i,PredLabels[i]]


    currentInterval[1] = len(PredLabels)-1
    intervals.append(currentInterval)
    
    for interval in intervals:
        #print(interval)
        
        labelsRight = TrueLabels[interval[0]:interval[1]+1] == interval[2]
        right = float(np.count_nonzero(labelsRight))/float(len(labelsRight))
        
        #print(TrueLabels[interval[0]:interval[1]+1].shape)
        #print(PredLabels[interval[0]:interval[1]+1])
        #print(interval[2])
        #print(labelsRight.shape)
        #print(right)
        #print("__________________________________________")

        intervalLength = interval[1]+1-interval[0]

        if right > acceptanceThreshhold:
            #Predicted the right thing... What is that thing?
            if interval[2] == 1:
                TP += intervalLength
            else:
                TN += intervalLength
        else:
            #Prdiction was wrong
            if interval[2] == 1:
                FP += intervalLength
            else:
                FN += intervalLength
        
    return TP,FP,TN,FN

import matplotlib.pyplot as plt

def AUCScore(model,DataSet,device,numberOfThresholdsTest = 100,EvaluateRegularDataOnIntervals = False,maxDatapointsFromDF = 100):
    

    if DataSet.IsGeneratedFromClassificationDS():
        score = scoreForClassificationData
    else:
        if EvaluateRegularDataOnIntervals:
            score = scoreForRegularData
        else:
            score = scorePointwise



    Errors = [0]*len(DataSet.Data())
    Labels = DataSet.Labels()
    
    if usingCupy:
        Labels = copy(Labels)
        for i,label in enumerate(Labels):
            Labels[i] = np.array(label)

    maxDeviation = 0

    model = model.to(device)

    for i in range(0,len(DataSet.Data())):
        
        seq_true = DataSet.Data()[i].to(device)
        seq_pred = model(seq_true)
        
        seq_true = seq_true[0,:,:].to("cpu").detach().numpy()
        seq_pred = seq_pred[0,:,:].to("cpu").detach().numpy()
        
        if usingCupy:
            seq_true = np.array(seq_true)
            seq_pred = np.array(seq_pred)
        
        if DataSet.IsGeneratedFromClassificationDS():
            Errors[i] = np.mean(abs(seq_true-seq_pred))
            if Errors[i] > maxDeviation:
                maxDeviation = Errors[i]
        else:
            Errors[i] = np.mean(abs(seq_true-seq_pred),axis = 0)
            if max(Errors[i]) > maxDeviation:
                maxDeviation = max(Errors[i])
    
    threshholds = np.linspace(0,maxDeviation,numberOfThresholdsTest)
    
    #Flippint the array, so the FPR and TPR are ascending order, (like on qould intuitively assume)
    threshholds = np.flip(threshholds)

    TPR=np.zeros(len(threshholds))
    FPR=np.zeros(len(threshholds))

    for i,t in enumerate(threshholds):

        TP = 0 #True Positive
        FP = 0 #False Positive
        TN = 0 #True Negative
        FN = 0 #False Negative

        for j,e in enumerate(Errors):

            resultTP,resultFP,resultTN,resultFN = score(Labels[j],e,t)
            TP += resultTP
            FP += resultFP
            TN += resultTN
            FN += resultFN
        

        if TP+FN == 0:
            TPR[i] = 0
        else:
            TPR[i] = float(TP)/float(TP+FN)
        

        if FP+FN == 0:
            FPR[i] = 0
        else:
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
    
    if usingCupy:
        return AUC.get(),errAUC.get()
    else:
        return AUC,errAUC


if __name__ == "__main__":

    print("Test Routine :)")

    print("Testing Regular Score:")

    def testRegScore(t,p):
        print(t)
        print(p)
        print(AdnaneScore(t,p))

    true1 = np.zeros(20)
    pred1 = np.zeros(20)
    pred1[7:15] = 1
    testRegScore(true1,pred1)
    print("Correct would be : TP = 0; FP = 1; TN =  2; FN = 0")

    true2 = np.zeros(20)
    true2[5:15] = 1
    pred2 = np.zeros(20)
    pred2[2:9] = 1
    pred2[11:15] = 1
    pred2[19] = 1
    testRegScore(true2,pred2)
    print("Correct would be : TP = 2; FP = 1; TN =  2; FN = 1")


