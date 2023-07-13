
![](../.mdpictures/Banners/ModelsBanner.png)

## Feed Forward Autoencoder

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

### Model Details

![](../.mdpictures/Models/RNN.png)

### Hyperparameters

<pre><code>
{
                "HiddenStates":32,
                "Layers":4,
                "BaseDecoderReconstructionOnZeros":True,
                "PerformLatentFlip":True, #flipping the latent space like malhorta et al.
                "CellKind":"LSTM"}
</code></pre>

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

