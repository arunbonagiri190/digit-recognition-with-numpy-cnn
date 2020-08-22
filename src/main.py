import numpy as np
from util import DataLoader,Activations
from conv_layer import Conv
from maxpool_layer import MaxPool
from dense_layer import Dense

def main():

    X, y = DataLoader.load_data()
    
    X_val = X[40001:]
    y_val = y[40001:]

    X = X[:40000]
    y = y[:40000]
    
    lr = 0.0001

    # Intializing Layers
    conv = Conv(5, (3,3), lr)
    pool = MaxPool()
    dense = Dense(845, lr)

    layers = [conv, pool, dense]

    fit(layers, X, y, ephocs=3)

    for pred_i in range(5):
        print("prediction: ",np.argmax(predict(layers, X_val[pred_i]))," | actual: ",np.argmax(y_val[pred_i].T))
    
    predict_for_kaggle_test_set(layers, "numpy_cnn_submission.csv")
    

def fit(layers, X, y, ephocs=1):
    """
    function for train the network

    Parameters:
        layers : list
        X      : numpy array
        y      : numpy array
        ephocs : integer
    
    Returns:
        (no-returns)
    """

    batch_size = X.shape[0]

    for ephoc in range(ephocs):

        te = []
        for i in range(0,batch_size):

            x0 = X[i]
            y0 = y[i].T
            
            # feedforward
            outs = forward(layers, x0)
            
            # loss calcs
            error = y0 - outs[2]
            te.append(error)

            # backpropagation
            backward(layers, error)
        
        print("ephoc ",(ephoc+1)," ...   error: ",np.mean(np.abs(sum(te)/batch_size)))
        

def forward(layers, x):
    """
    function for performing forward propagation in all the layers

    Parameters:
        layers : list
        x      : numpy array
    
    Returns:
        list : (contains output of all layers in the form of numpy arrays)
    """
    
    conv = layers[0]
    pool = layers[1]
    dense = layers[2]

    conv_output = conv.forward((x/255)- 0.5)
    pool_output = pool.forward(conv_output)
    dense_output = dense.forward(pool_output)

    return [conv_output, pool_output, dense_output]


def backward(layers, error):
    """
    function for performing backpropagation in all the layers

    Parameters:
        layers : list
        error  : numpy array
    
    Returns:
        (no-returns)
    """

    conv = layers[0]
    pool = layers[1]
    dense = layers[2]

    pool_error = dense.backward(error)
    conv_error = pool.backprop(pool_error)
    conv.backprop(conv_error)

def predict(layers, x):
    """
    function for predicting digit for given input

    Parameters:
        layers : list
        x      : numpy array
    
    Returns:
        outs[2] : numpy array (a one-hot-encoded digit)
    """

    outs = forward(layers, x)

    return outs[2]

def predict_for_kaggle_test_set(layers,filename):
    """
    this function is responsible for saving test predictions to given filename.
    
    Parameters:
        nn      :   object          
        filename:   str             
    Returns:
        (no-returns)
    """
    
    kaggle_test_set = DataLoader.load_test()
    preds = []

    for i in kaggle_test_set:
        preds.append(np.argmax(predict(layers, i)))

    DataLoader.save_predictions(preds, filename)

if __name__ == "__main__":
    main()