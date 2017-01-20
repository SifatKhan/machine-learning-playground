import pandas as pd
import numpy as np


class PassengerList:
    def __init__(self,data_frame,ticketlist,surnamelist,testdata=False):
        assert type(data_frame) is pd.DataFrame, "data_frame is not of type pandas.DataFrame: %r" % data_frame
        self._testdata = testdata
        self.data_frame = data_frame
        self.ticketlist = ticketlist
        self.ticketdict = dict(zip(ticketlist,range(ticketlist.shape[0])))
        self.surnamedict = dict(zip(surnamelist, range(surnamelist.shape[0])))
        self._fill_missing_ages_with_median()
        self._fill_missing_fares_with_median()
        self._convert_embarkport_to_int()
        self._convert_sex_to_gender_int()
        self._preprocess_cabins()
        self._traveling_w_family()
        self._preprocess_titles()

        self.data = self._get_data()
        

        ticketVector = np.zeros((self.data.shape[0],929))
        surnameVector = np.zeros((self.data.shape[0], len(surnamelist)))


        for idx in range(self.data.shape[0]):
            ticketidx = self.ticketdict[self.data[idx,6]]
            ticketVector[idx,ticketidx] = 1

            surnameidx = self.surnamedict[self.data[idx,25]]
            surnameVector[idx,surnameidx] = 1



        self._drop_uninteresting_columns()
        self.data = np.hstack((self._get_data(),ticketVector))
        self.data = np.hstack((self.data, surnameVector))




    def _get_data(self):

            if(self._testdata == False):
                data = self.data_frame.values[:,2:]
            else:
                data = self.data_frame.values[:,1:]
            return data

    def get_survival_data(self):
        data =  self.data_frame.values[:,1:2]
        return data

    def get_ids(self):
        ids = self.data_frame.values[:,0]
        ids= np.asmatrix(ids).T
        return ids

    def _preprocess_titles(self):
        df = self.data_frame
        df['Title'] = df['Name'].str.extract('\w,\s*?(\w+)',expand=True)
        df['Surname'] = df['Name'].str.extract('(\w+),\s*?\w+', expand=True)
        df['Mother'] = 0
        df.loc[((df['Title'] == 'Mrs') & (df['Age'] >17) & (df['Parch'] >0)) ,'Mother'] =1

        df['Miss'] = 0
        df.loc[((df['Title'] == 'Miss')), 'Miss'] = 1

        df['Mrs'] = 0
        df.loc[((df['Title'] == 'Mrs')), 'Mrs'] = 1

        df['Mr'] = 0
        df.loc[((df['Title'] == 'Mr')), 'Mr'] = 1
        del df['Title']


    def _preprocess_cabins(self):
        df = self.data_frame
        decks = []
        cabin_names = pd.Series(df[df['Cabin'].notnull()]['Cabin'].values).unique()
        for cab in cabin_names:
            decks.append(cab[0])
        unique_decks = np.unique(decks)

        df['CabinInt'] = 64
        for deck in unique_decks:
            df.loc[((df.Cabin.str.startswith(deck))&(df['Cabin'].notnull())),'CabinInt'] = ord(deck)

    def _traveling_w_family(self):
        df = self.data_frame
        df['FamilySize'] = df['SibSp'] + df['Parch']
        #maxfamily_size = df.FamilySize.max()
        maxfamily_size = 12
        for i in range(maxfamily_size):
            column_name = 'Family'+str(i)
            df[column_name] = 0
            df.loc[df['FamilySize']==i,column_name] =1

        del df['FamilySize']


    def _fill_missing_ages_with_median(self):
        df = self.data_frame
        medianFemaleAge = df[df['Sex'] == 'female']['Age'].median()
        medianMaleAge = df[df['Sex'] == 'male']['Age'].median()

        df.loc[((df['Age'].isnull()) &(df['Sex']=='female')),'Age'] = medianFemaleAge
        df.loc[((df['Age'].isnull()) &(df['Sex']=='male')),'Age'] = medianMaleAge

        age_mean = df['Age'].mean()
        age_std = df['Age'].std()
        print("lol")
        df['Age'] = (df['Age'] - age_mean) / age_std
        #df.loc[df['Age'] >62,'Age'] = 62

        #df.loc[df['Fare'] > 100, 'Fare'] = 100

    def _fill_missing_fares_with_median(self):
        df = self.data_frame
        medianFirstClassFare = df[df['Pclass'] == 1]['Fare'].median()
        medianSecondClassFare = df[df['Pclass'] == 2]['Fare'].median()
        medianThirdClassFare = df[df['Pclass'] == 3]['Fare'].median()

        df.loc[((df['Fare'].isnull()) &(df['Pclass']== 1)),'Fare'] = medianFirstClassFare
        df.loc[((df['Fare'].isnull()) &(df['Pclass']== 2)),'Fare'] = medianSecondClassFare
        df.loc[((df['Fare'].isnull()) &(df['Pclass']== 3)),'Fare'] = medianThirdClassFare

        df.loc[((df['Fare'] == 0) &(df['Pclass']== 1)),'Fare'] = medianFirstClassFare
        df.loc[((df['Fare'] == 0) &(df['Pclass']== 2)),'Fare'] = medianSecondClassFare
        df.loc[((df['Fare'] == 0) &(df['Pclass']== 3)),'Fare'] = medianThirdClassFare

        fare_mean = df['Fare'].mean()
        fare_std = df['Fare'].std()
        df['Fare'] = (df['Fare'] - fare_mean) / fare_std


    def _convert_sex_to_gender_int(self):
        self.data_frame['GenderInt'] = self.data_frame['Sex'].map( {'female': 1, 'male':0})

    def _convert_embarkport_to_int(self):
        self.data_frame['EmbarkedInt'] = self.data_frame['Embarked'].map( {'Q': 0, 'S':1, 'C':2, None: 2  })


    def _drop_uninteresting_columns(self):
        del self.data_frame['Ticket']
        del self.data_frame['Cabin']
        del self.data_frame['Name']
        del self.data_frame['Sex']
        del self.data_frame['Embarked']
        del self.data_frame['Parch']
        del self.data_frame['SibSp']
        del self.data_frame['Surname']


def read_csv(filename,ticketList,surnames,testlist=False):
    df = pd.read_csv(filename, header=0)
    passenger_list = PassengerList(df,ticketList,surnames,testlist)
    return passenger_list