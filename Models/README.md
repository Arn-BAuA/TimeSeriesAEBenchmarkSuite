
![](../.mdpictures/Banners/ModelsBanner.png)

The models in this framework all implement the same interface. They are subclasses of block, to handle the hyperparamters. They all take said hyperparameters, the dimension of the input stream and the compute device as constructor input, and they all are also subclasses of torch.nn.module.<br>
In the forward function, that the nn.Module objects have to overwrite in order to define the calculation done in the module, they all take a time series as an input and return a time series of similar dimensionality as output.<br>
In the next few sections we describe the diffrent models in this package in detail.

## Feed Forward Autoencoder

The feed forward autoencoder is the most generic reconstructing model that is available in the library. It consists of layers of feed forward perceptrons with a user specified activation function. The layer size does not have to conform to the classic auto encoder scheme (starting huge, getting smaller until the latent space is reached, getting larger again.). The first and last layer have the same number of neurons, one for each datapoint in the in and output. The user has two diffrent ways to configure the inner layers of the autoencoder.
* The user can pass an array of dubles, that contains the size of the inner layers in fractions of the input. An array like [0.5, 0.1, 0.5] would result in an autoencoder with 3 inner layers, the first has 0.5 times the number of neurons as the input, the second has 0.1 times the number of neurons and so on.
* The second way requires less hyper parameters. In this setting, the user just specifies the number of layers in the encoder and decoder and the size of the latent space layer. The constructor than linearly interpolates the size of the layers in between. An input of two layers per encoder/decoder and a latent space size of 0.5 for a input that has 64 valus would result in an autoencoder with 7 layers (1 input, two encoder, 1 latent space, 2 decoder, 1 output) that have the following number if neurons: (Input 64, Encoder 1: 56, Encoder 2: 48, Latent Layer: 32, Decoder 1: 48, Decoder 2: 56, Output 64).<br>

There are also two ways to feed the time series in the input layer of the encoder:
* One way is to  just flatten the time series, if it is multidimensional.
* The second way is to cut the time series into snippets, than flatten these snippets and than concatenate the input. The idea here is, that the parts in the different dimensions, that describe one interval in time are closer together, when des into the encoder. This strategy is not depicted in the picture showing the model details for the feed forward AE. It is depicted in the picture for the attention based method, since this is one of the word embedding methods that are at hand for this model. 

### Model Details

![](../.mdpictures/Models/FFAE.png)

### Hyperparameters

<pre><code>
return {"InputSize":150,
                "LayerSequence":[1,0.7,0.3,0.2,0.3,0.7,1],#in multiples of the input ... first and last must be 0
                "ActivationFunction":"ReLU",
                "SlicingBeforeFlatten":True,#If Slicing is true, the tensors will not be flattend, the tensors will be sliced along the time axis, these slices will than be flattend and concatenated.
                "SliceLength":0.1, #of sequence lenght
                "LayerSequenceLinear":True, #if this is true, the layer sequence above will be ignored. instead a linear progression of th layer size to the latent space will be calculated. The layer Sequence will than be overwritten by that.
                "LatentSpaceSize":0.1,
                "NumLayersPerPart":3
                } 
</code></pre>

## Recurrend Autoencoder

This model realizes an autoencoder using recurrend neural networks (either torch.nn.GRU or torch.nn.LSTM). The latent space is realized by creating two recurrend network stacks. The first stack processes the input time series sequentially. After this sequential reading is complete, the hidden states of this layer stack represent the latent space (the direct output is ignored). The hidden states get then transfered to the second stack, which gets a zero-vector as input and should rebuild the input based on the hidden states.<br>
Optionally, we added the posibillity to flip the latent hidden states and, if they where flipped, flip the output later. This is a strategy that was adopted from Malhotra et al. (https://arxiv.org/pdf/1607.00148.pdf). The idea here is, that by flipping, the second RNN stack constructs the time series from the last processed point on into the past, which is more natural from the flow of information, than starting with the first point in time (which was also the first one processed by the first stack, so the resemblens of it in the hidden states might be the smallest) to the last.

### Model Details

![](../.mdpictures/Models/RNN.png)

### Hyperparameters

Cell kind can have the two values "LSTM" and "GRU".

<pre><code>
{
                "HiddenStates":32,
                "Layers":4,
                "BaseDecoderReconstructionOnZeros":True,
                "PerformLatentFlip":True, #flipping the latent space like malhorta et al.
                "CellKind":"LSTM"}
</code></pre>

### Some Notes

When training the network with mostly default settings on the batched trainer (batch size = 10, learning rate in the adam optimizer = 0.001) we noticed, that the loss "jumps" when plotted against the epochs. The loss function remains stationary for many epochs, and than suddently reaches another value. This new value is in some cases higher, in some cases lower then the previous one. We do not know if this behaviour can be omitted by fine tuning the learning rate or a diffrent batching strategy.<br>
As a hot fix for this, it may be advantageous to include a history for the training, like for the LSTM based approach that is shown in this blog (our very first  prototype implementation of this library  was based on it and we kept the history functionality ever since... https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/).

## Convolutional Autoencoder

### Model Details

![](../.mdpictures/Models/CNN.png)

### Hyperparameters

<pre><code>
        return {
                "InputSize":150,#size of the input data...
                "KernelSize":5,#Size of kernel for the convolution
                "LayerSequenceEnc":[0.7,0.5], #Size of the layers in relation to the input
                "LatentSize":0.2, #size of the latent space in relation to the input
                "LayerSequenceDec":[0.5,0.7], #size of the layer in relation to the input
                "hasFFTEncoder":False, #if true, a second encoder, beside the time encoder is added, that porcesses the fft. 
                "HanningWindowBeforeFFT":True, #Uses the Hanning window before computing the FFT, if FFT encoder is there...
                "GlueLayerSize":2, #There is a stack of perceptrons that takes the output of the FFT- and TimeDecoder and Broadcasts them to the encoder input. That stack can have a height, specified here.
                "hasOnlyFFTEncoder":False,#if true, only a fft encoder is provided
                "ActivationFunction":"ReLU", #activation function used perceptrons across the net
               # "DownsampleByPooling":True, #if true, the downsampling in the decoder is done by pooling. if not, a layer of perceptrons does the job.
                "BatchNorm":True,#if true, a batchnorm is applied after each covolution.
            }
 
</code></pre>

## Attention Based Autoencoder

### Model Details

![](../.mdpictures/Models/AttentionBased.png)

### Hyperparameters

<pre><code>
{"InputSize":150,
                "nWords":15,
                "WordSize":20,
                "nPreprocessorLayers":2,
                "ActivationFunctionPreprocessor":"Tanh",
                "FeedDirect":True,#if false, a linear feed forward network will be used instead of the embedding
                "nAttentionHeads":4,
                "nTrEncoderLayers":2,
                "nTrDecoderLayers":2,
                "TrFFDim":40,
                "nPostprocessorLayers":4,
                "ActivationFunctionPostprocessor":"Tanh",}
</code></pre>

