
from torch import nn

class Model(nn.Module):#Autoencoder form Bin's drift detection script. (Malhorta et al)
    
    def __init__(self,device,**HyperParameters):
        
        defaultHyperParameters = 
            {
                "Dimensions":1,
                "HiddenStates":32,
                "Layers":4
            }

        HPs = {**defaultHyperParameters, **HyperParameters}

        super().__init__()
        self.input_dim,self.hidden_dim = HPs["Dimensions"],HPs["HiddenStates"]

        self.encoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = HPs["Layers"],
                    batch_first = True
                )

        self.decoder = nn.LSTM(
                    input_size = self.input_dim,
                    hidden_size = self.hidden_dim,
                    num_layers = HPs["Layers"],
                    batch_first = True
                )
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.output = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self,x):
        
        encoder_out,encoder_hidden = self.encoder(x)
        
        encoder_hidden = torch.flip(encoder_hidden,[1])

        decoder_input = torch.zeros_like(x)
        
        decoder_output,decoder_hidden = self.decoder(decoder_input,encoder_hidden)
        reconstruction = self.output(decoder_output)
        reconstruction = torch.flip(reconstruction,[1])
        return reconstruction


