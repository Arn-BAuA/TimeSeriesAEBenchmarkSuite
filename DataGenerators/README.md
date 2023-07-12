
![](../.mdpictures/Banners/DataGeneratorsBanner.png)

At the moment, there is only one data generator. The process how data is generated in this instance can be generalized to create a whole family of data generators (Thats a to do for the future.).<br>
A data generator in this framework is always a function that takes a set of hyperparameters and outputs training-, validation- and test-set in the data block format that have to be used in the framework.

## How do the data generators work:

In this section, we introduce how the data generators in this framework work on the example of the Sines.py data generator.<br>
The data generator has the possibility to create multivariate time series data. Each dimension in the dataset is filled with a superposition of sine functions.<br>
In our implementation, the sine functions are described by three parameters. The amplitude, the frequency and the offset (we don't include a phase). For each of these parameters, a normal and an abnormal  range is specified. Data is generated as follows: A random walk in the normal and in the abnormal parameter space is simulated. The user can specify a maximum speed in the parameter space at which this random walk occurs. <br>
For each point in time for the random walk, it is also decided, how much of the abnormal and how much of the normal random walk should be concidered. This array is called the blend array. Usually it is filled with zeros and ones and a vew values between the changes to aid with the transition. <br>
The normal and the abnormal random walk are then blended according to the values of the blend array. This set of parameters, that is obtained this way for each point in time is then used as an input for the sine function. Each of the single sine functions in each of the dimensions is generated this way. They share a common blend array tough.<br>
The data snippets for the three data sets are sampled form this huge time series afterwards.

## The Hyperparameters of the Sines datagenerator:

We will discuss the hyper parameters of the sine generator in depth, since they can be a bit confusing and there is quiet a vew of them.<br>
The default parameters are split into small sections (the code). The meaning of each section is discussed below.

<pre><code>
    defaultHyperParameters = {
        "Amplitudes" :          [[1],[1] ],#center of the amplitude parameter span
        "AmplitudeSpan" :       [[0.2],[0.1] ],#plus minus
        "AnomalousAmplitudes" :     [[1.6],[1.6]], 
        "AnomalousAmplitudeSpan" :  [[0],[0]], #Plus Minus
        
        "Frequency" :           [[0.2],[0.3]],
        "FrequencySpan" :       [[0],[0]], #Plus Minus
        "AnomalousFrequency" :      [[0.5],[0.5]], 
        "AnomalousFrequencySpan" :  [[0],[0]],#Plus Minus
        
        "Offset" :              [[0],[0]],
        "OffsetSpan":           [[0],[0]],
        "AnomalousOffset" : [[0],[0]], #Plus Minus
        "AnomalousOffsetSpan" : [[0],[0]], #Plus Minus

</pre></code>

These are the boundaries of the normal and anomal area of the random walk in the parameter space.<br>
The paramters are passed as a two dimensional array. The first array has one array for each dimension of the time series.
We thus can see, that the default hyper parameters describe two dimensions. The sub arrays in this array contain numbers.
Each number represents one sine function that gets superpositioned to form the values that are stored in that dimension. With the default hyper parameters, we can see, that there is only a single sine function per dimension.<br>
In addition, the user can also demand a number of dimensions right at the call of the data generator. When this number is 
smaller than the number of dimensions described in the hyper parameter array, only the first elements will be used. If the number is larger, multiple dimensions will be greated form the same parameter set arrays.<br>
The Parameters are given as center and span. Looking at Amplitudes and Amplitude span, we can see that the first described dimension consists of a sine with amplitude values between 0.8 and 1.2 (1 +/- 0.2). The Amplitude in the anomalous case is always 1.6 (1.6 +/- 0). In the second dimension, the normal amplitude varies between 0.9 and 1.1 (1 +/- 0.1) and so on and so forth...

<pre><code>
        "AnomalyMagnitudeInAmplitude" :      [[1],[1]],
        "AnomalyMagnitudeInFrequency" :      [[0.1],[0.1]],
        "AnomalyMagnitudeInOffset"    :      [[0.1],[0.1]],
        "AnomalyInDimension" : [[1],[1]],
</pre></code>

The parameters above can be used to further control how anomalies manifest.
Anomalies are created by blending the normal parameter space random walk with the anomal one. The MagnitudeIn... parameters are maximum values for this blend. We can read out, that, if an anomalie occures, the blend of the amplitude will consist of only anomalous values (1 \* anomaly + 0 \* normal data). The frequencies and offsets are blended in more subtle (0.1 \* anomalie + 0.9 \* normal).<br>
With the anomaly in dimension array, it can be specified which of the sines should be anomal. The notation here is the same as with the parameter space above. In this example, the array states that both sines that are described are affected by the anomaly. 0 would mean, that both are not affected.

<pre><code>
        "NoiseLevel" :              0.02,
</pre></code>

Uniform distributed noise of the specified amplitude is added to all time series.

<pre><code>
        "MaxSystemVelocity":           0.01, # The maximum rate at which the parameters change in parameter value per dimension per second.
        "SystemChangeRate": 0.01,
</pre></code>

The random walk in the parameter space is created by randomly selecting points in the parameter space and linear interpolation between these points. The System change rate states, how many points per unit time are generated. The MaxSystemVelocity is the maximum speed of the random walk in parameter space.

<pre><code>
        "AnomaliesInTrainingdata" : False,
        "AnomaliesInValidationdata" :True,
        "ContinousFrequencyBlending":True, #..
</pre></code>

Test data always contains anomalies. With the first two values, the user can decide, if anomalies should also occure in training and validation set.<br>
Continous frequency blending: If the frequency would just be normally blended and than multiplied by the time and put into the sine wave, this would lead to strange artifacts during the blending. Using continous frequncy blending, the product of time and frequency rather than the frequency will be blended, leading to a smooth transition.

<pre><code>
        "AnomalyChance":            0.006,
        "AnomalyRampTime":          1,
        "AnomalyDuration":   2,
        "AnomalyThreshold": 0.3,
</pre></code>

The anomaly chance is the rate at which anomalies occur.
Anomaly duration is the duration of the anomalies, Ramp time is the time that the transition from normal to anomal parameter space takes. Anomaly threshold is a parameter that is needet to lable the data. It states, how abnormal the value in the blend array must be for a point to be marked as anomal. In the currend example, the anomaly takes one unit of time to fully occur, but at that point, where the ramp exceedes a value of 0.3, the values will be marked as anomaly.

<pre><code>
        "SampleTime":               0.2,
        "Duration":                 2000,#duration of the timeseries in the dataset 
        "RandomSeed":               1,
        "SampleWindowSize":150,
</pre></code>

These arguments are importand for the sampling of the created tim series.
Sample time is the duration between two samples. In this example, the generated tme series in sampled every 0.2 units, making a sample rate of 5.<br>
Duration is the duration of the time series that will be created to cut the snippets for the sets.<br>
The random seed parameter was created to control, if a fixed seed, or a random seed should be used for the random walk. However, this is not implemented at the moment.<br>
SampleWindowSize is the number of data points per time series snippet that is used in the data sets.

<pre><code>
        "TrainingSetSize": 200,
        "ValidationSetSize":100,
        "TestSetSize" : 30,
    }
</pre></code>

These Parameters can be used to determine the number of time series snippets per data set.
