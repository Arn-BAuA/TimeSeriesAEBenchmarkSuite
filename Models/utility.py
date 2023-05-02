
activations = {
            "ReLU":torch.nn.ReLU,
            "Sigmoid":torch.nn.Sigmoid,
            "tanh":torch.nn.Tanh
        }

#Method for converting the arguments passed 
# in the Hyperparameters as string to the
#actual pytorch methods.
def strToActivation(string):
    return activations[string]
    

