from sklearn.preprocessing import LabelEncoder
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
#import model

class MultiLabelEncoder():
    """Classe gérant plusieurs LabelEncoder en même temps."""
    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def set_params(self, labelencoders = {}):
        self.labelencoders = labelencoders
        #print('MLE - found le : ', labelencoders.keys())
        if len(self.labelencoders) == 0 : 
            self.fitted = False
        else :
            self.fitted = True
        print('MLE - fitted ?', self.fitted)
        
    def get_params(self, deep = False):
        return {}

    def fit(self, X, y=None):
        try :
            cols = list(X.columns)
        except :
            cols = list(range(len(X)))
        if self.fitted == False :
            print('MLE - fitting')
            x = np.array(X)
            for i in range(x.shape[1]):
                le = LabelEncoder()
                le.fit(x[:, i])
                #self.labelencoders.append(le)
                self.labelencoders[cols[i]] = le
            self.fitted = True

    def transform(self, X, y=None):
        try :
            cols = list(X.columns)
        except :
            cols = list(range(len(X)))
        x = np.array(X)
        new_X = np.zeros(x.shape)
        for i in range(x.shape[1]):
            try :
                le = self.labelencoders[cols[i]]
                new_X[:, i] = le.transform(x[:, i])
            except : 
                print('Error', i, cols[i])
                le = self.labelencoders[cols[i]]
                new_X[:, i] = le.transform(x[:, i])
        return new_X

    def inverse_transform(self, X):
        x = np.array(X)
        new_X = np.zeros(x.shape, dtype=object)
        for i in range(x.shape[1]):
            print(i)
            le = self.labelencoders[i]
            new_X[:, i] = le.inverse_transform(np.int64(x[:, i]))
        return new_X

    def fit_transform(self, X, y=None):
        self.fit(X)
        new_X = self.transform(X)
        return new_X