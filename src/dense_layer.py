import numpy as np
from util import Activations

class Dense:

    nodes_in = 0
    nodes_out = 10
    lr = 0.0

    weights, biases = np.array([]), np.array([])
    z, a = np.array([]), np.array([])

    nWeights, nBiases = np.array([]), np.array([])

    last_input = np.array([])
    pool_out_shape = []
    check = False
    

    def __init__(self, nodes, lr):
        """
        Parameters:
            nodes : integer
            lr    : float
    
        """
        self.nodes_in = nodes
        self.lr = lr

        self.weights = np.random.randn(self.nodes_out, self.nodes_in)/np.sqrt(self.nodes_out)
        self.biases = np.random.randn(1, self.nodes_out)
    
    def forward(self, x):
        """
        function for performing forward propagation

        Parameters:
            x : numpy array
        
        Returns:
            self.a : numpy array
        """
        
        self.pool_out_shape = x.shape
        
        x = x.flatten()
        x = x.reshape(x.shape + (1,)).T
        self.last_input = x

        self.z = np.dot(x, self.weights.T) + self.biases
        self.a = Activations.softmax(self.z)

        if(self.check):
            print("self.pool_out_shape: ",self.pool_out_shape)
            print("self.last_input.shape: ",self.last_input.shape,"\nself.last_input: ",self.last_input)
            print("\nself.z.shape : ",self.z.shape," self.z: ",self.z)
            print("\nself.a.shape : ",self.a.shape," self.a: ",self.a)

        return self.a

    def backward(self, error):
        """
        function for performing backpropagation in all the layers

        Parameters:
            error  : numpy array
        
        Returns:
            pool_error : numpy array
        """
        # print("before self.weights: ",self.weights)
        # print("before self.biases: ",self.biases)

        self.nBiases = error
        self.nWeights = error.T.dot(self.last_input)


        pool_error = error.dot(self.weights)
        pool_error = pool_error.reshape(self.pool_out_shape)

        self.weights += (self.lr * self.nWeights)
        self.biases  += (self.lr * self.nBiases)

        if(self.check):
            
            print("error.shape: ",error.shape)
            print("self.nBiases shape: ",self.nBiases.shape)
            print("self.nWeights shape: ",self.nWeights.shape)
            print("pool_error.shape: ", pool_error.shape)

            print("after self.weights: ",self.weights)
            print("after self.biases: ",self.biases)

        return pool_error
        
    

        