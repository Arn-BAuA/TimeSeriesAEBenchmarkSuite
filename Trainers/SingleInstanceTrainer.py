##
# This training routine is "borrowed" from a tutorial on couriously.com

from torch import optim
import copy


class Trainer():

    def __init__(self,model,**hyperParameters):

        defaultHyperParameters = 
        {

        }

        HPs = {**defaultHyperParameters,**hyperParameters}

        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.criterion = nn.L1Loss(reduction = "sum").to(device)

        best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = 1e9

def doEpoch(self,model,trainingSet,validationSet,history,performanceGoals=[],device):
    
    train_loss = []
    for seq_true in trainingSet:

        self.optimizer.zero_grad()
        
        seq_true = seq_true.to(device) # I think we can do this faster if we transfer the dataset to the gpu first and than do the training
        seq_pred = model(seq_true)
    
        loss = self.criterion(seq_pred, seq_true)

        loss.backward()
        self.optimizer.step()

        train_loss.append(loss.item())

    val_loss = []

    model = model.eval()
    
    with torch.no_grad():
        for seq_true in validationSet:

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = self.criterion(seq_pred,seq_true)
            val_loss.append(loss.item())

    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if val_loss < best_loss:
        self.best_loss = self.val_loss
        self.best_model_wts = copy.deepcopy(model.state_dict())

    return model,history

