import pandas as pd
import numpy as np
import os.path
import math

class HouseList:
    def __init__(self,trainfile,testfile,use_validationset=False):
        assert os.path.isfile(trainfile) is True, "Training data file %r does not exist" % trainfile
        assert os.path.isfile(testfile) is True, "Testing data file %r does not exist" % testfile
        assert type(use_validationset) is bool, "use_validationset %r is not of type bool" % testfile
        self._train_dataframe = pd.read_csv(trainfile)
        self._test_dataframe = pd.read_csv(testfile)

        self._test_dataframe.loc[666, "GarageQual"] = "TA"
        self._test_dataframe.loc[666, "GarageCond"] = "TA"
        self._test_dataframe.loc[666, "GarageFinish"] = "Unf"
        self._test_dataframe.loc[666, "GarageYrBlt"] = 1980

        self._test_dataframe.loc[1116, "GarageType"] = np.nan

        self._train_dataframe = self._train_dataframe[self._train_dataframe["GrLivArea"] <= 4000]
        self._train_dataframe = self._train_dataframe[self._train_dataframe["SalePrice"] <= 400000]

        self._add_means_to_nulls()

        self.test_ids = np.asmatrix(self._test_dataframe.select_dtypes(exclude=['object']).values[:, 0]).T
        self.testdata = self._test_dataframe.select_dtypes(exclude=['object']).values[:, 1:]
        self.traindata = self._train_dataframe.select_dtypes(exclude=['object']).values[:, 1:-1]
        self.trainlabels = np.asmatrix(self._train_dataframe.select_dtypes(exclude=['object']).values[:, -1]).T

        ## Get all the possible non-numeric columns
        train_string_columns = self._train_dataframe.select_dtypes(include=['object']).columns
        test_string_columns = self._test_dataframe.select_dtypes(include=['object']).columns
        all_string_columns = np.unique(np.hstack((train_string_columns,test_string_columns)))


        ## We are going to build a one-hot vector for every non-numerical column. The size of the
        ## one-hot vector is going to depend on how many possible values each column has.
        test_onehot_vectors = []
        train_onehot_vectors = []

        for string_col in all_string_columns:

            print("Processing column: %r" %string_col)

            ## Assign the string 'NA' to all null rows.
            self._train_dataframe.loc[self._train_dataframe[string_col].isnull(),string_col] = 'NA'
            self._test_dataframe.loc[self._test_dataframe[string_col].isnull(),string_col] = 'NA'

            ## Get all the possible string values for this column
            train_values = self._train_dataframe[string_col].unique()
            test_values = self._test_dataframe[string_col].unique()
            all_values = np.unique(np.append(train_values,test_values))

            value_dict = dict(zip(all_values,range(all_values.shape[0])))

            train_onehot_vector = np.zeros((len(self._train_dataframe),len(all_values)))
            test_onehot_vector = np.zeros((len(self._test_dataframe),len(all_values)))

            ## This could be done better I believe. This is very slow...
            ## ... and hacky
            idx = 0
            for index, row in self._train_dataframe.iterrows():
                value = row[string_col]
                value_idx = value_dict[value]
                train_onehot_vector[idx, value_idx] = 1
                idx+=1

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

        validationset_size = 250

        self.validationdata = self.traindata[-validationset_size:]
        self.validationlabels = self.trainlabels[-validationset_size:]

        if(use_validationset == True):
            self.traindata = self.traindata[0:-validationset_size]
            self.trainlabels = self.trainlabels[0:-validationset_size]




    def _add_means_to_nulls(self):
        
        df = self._train_dataframe
        tdf = self._test_dataframe
        
        test_null_cols = tdf.drop(['Id'],axis=1).select_dtypes(exclude=['object']).isnull().columns
        train_null_cols = df.drop(['Id','SalePrice'],axis=1).select_dtypes(exclude=['object']).isnull().columns


        
        for null_col in train_null_cols:
            median = df[null_col].median()
            #median = 0.0
            df.loc[(df[null_col].isnull()),null_col] = median
            tdf.loc[(tdf[null_col].isnull()),null_col] = median

