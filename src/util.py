import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class DataLoader:

    path = '../data/'

    # function for loading data from disk
    @classmethod
    def load_data(self):
        """
        this function is responsible for loading traing data from disk.
        and performs some basic opertaions like
            - one-hot encoding
            - feature scaling
            - reshaping data

        Parameters:
            (no-parameters)

        Returns:
            X   :   numpy array         
            y   :   numpy array         
        
        """
        
        if(not Path(self.path+'train.csv').is_file()):
            print("[util]: train data not found at '",self.path,"'")
            #quit()

        print("[util]: Loading '",self.path+'train.csv',"'")
        train_df = pd.read_csv(self.path+'train.csv')

        y = np.array(pd.get_dummies(train_df['label']))

        X = train_df.drop(['label'], axis=1)
        X = np.array(X)

        X = X.reshape(X.shape[0],28,28)
        y = y.reshape(y.shape + (1,))
        del train_df

        return X, y
    
    @classmethod
    def load_test(self):
        """
        this function is responsible for loading test data from disk.
        and performs - reshaping data

        Parameters:
            (no-parameters)

        Returns:
            test_x   :   numpy array              
        
        """

        if(not Path(self.path+'test.csv').is_file()):
            print("[util]: test data not found at '",self.path,"'")
            #quit()

        print("[util]: Loading '",self.path+'test.csv',"'")
        test_df = pd.read_csv(self.path+'test.csv')
        
        test_x = np.array(test_df)
        test_x = test_x.reshape(test_x.shape[0],28,28)
        del test_df
        
        return test_x
    
    # custom function for saving kaggle test data predictions
    @classmethod
    def save_predictions(self, preds, filename='new_submission.csv'):
        """
        this function is responsible for saving test predictions to given filename.
        
        Parameters:
            preds   :   numpy array     (all the predictions of test set)
            filename:   str             (filename for saving & identifying different test predictions)

        Returns:
            (no-returns)
        """
        sub_path = self.path+'sample_submission.csv'
        
        if(not Path(sub_path).is_file()):
            print("[util]: sample_submission file not found at '",sub_path,"',\n\t it is required to get submission format")

        submission = pd.read_csv(sub_path)
        submission['Label'] = preds
        submission.to_csv(self.path+filename, index=False)

class Activations:
    
    # sigmoid activation function with derivative
    @classmethod
    def sigmoid(self, x, derivative=False):
        if(derivative):
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        
        return 1.0/(1.0 + np.exp(-x))


    # relu activation function with derivative
    @classmethod
    def relu(self, x, derivative=False):
        if(derivative):
            return x > 0
        
        return np.maximum(x, 0)


    # softmax activation function
    @classmethod
    def softmax(self, z):
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1)