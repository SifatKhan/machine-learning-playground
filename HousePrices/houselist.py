import pandas as pd
import numpy as np
import os.path
import math

class HouseList:
    def __init__(self,trainfile,testfile):
        assert os.path.isfile(trainfile) is True, "Training data file %r does not exist" % trainfile
        assert os.path.isfile(testfile) is True, "Testing data file %r does not exist" % testfile
        self._train_dataframe = pd.read_csv('train.csv')
        self._test_dataframe = pd.read_csv('test.csv')

        self.testdata = self._test_dataframe.select_dtypes(exclude=['object']).values[:, 1:]
        self.traindata = self._train_dataframe.select_dtypes(exclude=['object']).values[:, 1:]

        train_string_columns = self._train_dataframe.select_dtypes(include=['object']).columns
        test_string_columns = self._test_dataframe.select_dtypes(include=['object']).columns

        # get all the possible non-numeeric columns
        string_columns = np.unique(np.hstack((train_string_columns,test_string_columns)))

        test_onehot_vectors = []
        train_onehot_vectors = []

        for string_col in string_columns:

            print("Processing column: %r" %string_col)
            self._train_dataframe.loc[self._train_dataframe[string_col].isnull(),string_col] = 'NA'
            self._test_dataframe.loc[self._test_dataframe[string_col].isnull(),string_col] = 'NA'

            train_values = self._train_dataframe[string_col].unique()
            test_values = self._test_dataframe[string_col].unique()
            all_values = np.unique(np.append(train_values,test_values))

            value_dict = dict(zip(all_values,range(all_values.shape[0])))

            train_onehot_vector = np.zeros((len(self._train_dataframe),len(all_values)))
            test_onehot_vector = np.zeros((len(self._test_dataframe),len(all_values)))

            for idx in range(len(self._train_dataframe)):
                value = self._train_dataframe[string_col][idx]
                value_idx = value_dict[value]
                train_onehot_vector[idx,value_idx] = 1

            for idx in range(len(self._test_dataframe)):
                value = self._test_dataframe[string_col][idx]
                value_idx = value_dict[value]
                test_onehot_vector[idx,value_idx] = 1

            test_onehot_vectors.append(test_onehot_vector)
            train_onehot_vectors.append(train_onehot_vector)

        train_vectors = np.hstack(train_onehot_vectors)
        test_vectors = np.hstack(test_onehot_vectors)
        self.traindata = np.hstack((self.traindata,train_vectors))
        self.testdata = np.hstack((self.testdata, test_vectors))













