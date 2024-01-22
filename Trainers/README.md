
![](../.mdpictures/Banners/TrainersBanner.png)

The Trainer is a class that encapsulates the training process of the model, so that it can be used in a modular fashon. The trainer basically takes training and validation set from one data source and a model and holds the methods to train that model on that data. All Trainer models must implement the following interface:<br>

* The trainer is a subclass of the block, to handle the hyperparameters.
* The trainer takes the model and the computedevice("cpu" / "gpu") in the constructor.
* The trainer has a method called setDataSets. It is called in Benchmark.py before the trainine and can be used to implement data augmentation strategies.
* The trainer has a doEpoch method, that gets called in Benchmark.py to perform one epoch of training on the model and return said model.
* The trainer has a finalizeTraining method that is called after the training to do final touches on the model if necessairy (the batched trainer e.g. uses it.)

## Batched Trainer:
At the moment the only supported trainer is the batched trainer. It encapsulates the functionality of all previous iterations of trainers this repository saw.<br>
The batched trainer is capable of grouping the passed data sets in batches of fixed number. The loss functions and training are than optimized on these batches. The idea here is that this presents an implicit regularisation due to the loss function beeing optimized on a bunch of examples, making optmisation less susceptible for special features that are only present by coincidence on one example.<br>
In addition to the batching, the batched trainer has a history function: It trains the model on the training set and evaluates its reconstruction loss on the validation set. If the error on the validation set is higher than the previously recorded minimal error, the model weights are swapped agianst the wights of this model. We picked this idea up form this blog post: https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/ . From our experience it hepls especially when training LSTMs.<br>
The batched trainer uses Adam Optimizer with a learning rate of 0.001.
 
