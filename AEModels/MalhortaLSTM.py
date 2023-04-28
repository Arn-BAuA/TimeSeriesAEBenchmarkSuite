import torch
from torch import nn
from BlockAndDatablock import block

class Model(block,nn.Module):#Autoencoder form Bin's drift detection script. (Malhorta et al)
    
    def _getDefaultHPs(self):
        return {
                "HiddenStates":32,
                "Layers":4}

    def __init__(self,Dimensions,device,**HyperParameters):

        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"MalhortaLSTM_AE",**HyperParameters)
        nn.Module.__init__(self)
        self.input_dim,self.hidden_dim = Dimensions,self.HP["HiddenStates"]

        self.encoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = self.HP["Layers"],
                    batch_first = True
                )

        self.decoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = self.HP["Layers"],
                    batch_first = True
                )
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.output = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self,x):
        
        encoder_out,(encoder_hidden,encoder_cell) = self.encoder(x)
        
        encoder_hidden = torch.flip(encoder_hidden,[1])
        encoder_cell = torch.flip(encoder_cell,[1])

        decoder_input = torch.zeros_like(x)
        
        decoder_output,decoder_hidden = self.decoder(decoder_input,(encoder_hidden,encoder_cell))
        reconstruction = self.output(decoder_output)
        reconstruction = torch.flip(reconstruction,[1])
        return reconstruction


