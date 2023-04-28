import torch
from torch import nn
from BlockAndDatablock import block

class Model(block,nn.Module):#Autoencoder form Bin's drift detection script. (Malhorta et al)
    
    def _getDefaultHPs(self):
        return {
                "HiddenStates":32,
                "Layers":4,
                "BaseDecoderReconstructionOnZeros":True,
                "PerformLatentFlip":True, #flipping the latent space like malhorta et al.
                "CellKind":"LSTM"} #can be "LSTM" or "GRU"

    def __init__(self,Dimensions,device,**HyperParameters):

        self.device = device
        self.Dimensions = Dimensions

        block.__init__(self,"Recurrend AE",**HyperParameters)
        nn.Module.__init__(self)
        self.input_dim,self.hidden_dim = Dimensions,self.HP["HiddenStates"]
        
        if self.HP["CellKind"] == "LSTM":
            self.isLSTM = True
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
        else:
            self.isLSTM = False  # is GRU
            
            self.encoder = nn.GRU(
                                  input_size = self.input_dim,
                                  hidden_size = self.hidden_dim,
                                  num_layers = self.HP["Layers"],
                                  batch_first = True
                                )

            self.decoder = nn.GRU(
                                  input_size = self.input_dim,
                                  hidden_size = self.hidden_dim,
                                  num_layers = self.HP["Layers"],
                                  batch_first = True
                                )
        
        
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.output = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self,x):
        #This has some doubled code... maybe there is a better way...
        if self.isLSTM:
            return self._LSTMForward(x)
        else:
            return self._GRUForward(x)

    def _LSTMForward(self,x):
        encoder_out,(encoder_hidden,encoder_cell) = self.encoder(x)
        
        if self.HP["PerformLatentFlip"]:
            encoder_hidden = torch.flip(encoder_hidden,[1])
            encoder_cell = torch.flip(encoder_cell,[1])
        
        if self.HP["BaseDecoderReconstructionOnZeros"]:
            decoder_input = torch.zeros_like(x)
        else:
            decoder_input = encoder_out

        decoder_output,decoder_hidden = self.decoder(decoder_input,(encoder_hidden,encoder_cell))
        reconstruction = self.output(decoder_output)
        
        if self.HP["PerformLatentFlip"]:
            reconstruction = torch.flip(reconstruction,[1])
        
        return reconstruction

    def _GRUForward(self,x):

        encoder_out,encoder_hidden = self.encoder(x)
        
        if self.HP["PerformLatentFlip"]:
            encoder_hidden = torch.flip(encoder_hidden,[1])

        if self.HP["BaseDecoderReconstructionOnZeros"]:
            decoder_input = torch.zeros_like(x)
        else:
            decoder_input = encoder_out
        
        decoder_output,decoder_hidden = self.decoder(decoder_input,encoder_hidden)
        reconstruction = self.output(decoder_output)
        
        if self.HP["PerformLatentFlip"]:
            reconstruction = torch.flip(reconstruction,[1])
        
        return reconstruction
