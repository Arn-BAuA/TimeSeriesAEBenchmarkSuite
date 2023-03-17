
from torch import nn

class Model(nn.Module): #Plain Feed Forward Encoder....
    
    def __init__(self,device,**HyperParameters):
        
        defaultHyperParameters = 
            {
                "Dimensions":1,
                "InputSize":125,
            }

        HPs = {**defaultHyperParameters, **HyperParameters}
        
        super().__init__()

        self.model = nn.Sequential(
                    torch.nn.Linear(HPs["Dimensions"]*HPs["InputSize"] , 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20 , 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50 , 100),
                    torch.nn.ReLU(),    
                    torch.nn.Linear(100,HPs["Dimensions"]*HPs["InputSize"])
                )

        self.model.to(device)

    def forward(self,x):
        x = torch.transpose(x,0,1)
        x = self.model(x)
        return torch.transpose(x,0,1)



