import pandas as pd 
import datetime
import pickle
import os
import glob
import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle, resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from plot_utils import plot_confusion_matrix


## general params:
LOAD_PRETRAINED_MODELS = True


class Dataset(object):
    '''
    this class is used to more easily structure classification datasets between the data (matrix of values) and the target (class)
    instead of doing any indexing when training an algorithm, simply use Dataset.data and Dataset.target
    '''
    def __init__(self, df, target_col_name):
        self.target_col_name = target_col_name
        self.target = df.loc[:, target_col_name]
        self.data = df.loc[:, df.columns != target_col_name]
    def __str__(self):
        return ('first 10 data points\n' + str(self.data.head(10)) + '\nfirst 10 labels\n' + str(self.target.head(10)) + '\n')

class MachineLearningModel(object):
    '''
    this is to help separate models by attributes, 
    i.e. this is a sklearn model and hence can use sklearn stuff, or a Keras model, or a XGBoost model, etc etc etc
    '''
    def __init__(self, model, model_type, framework, id=None):
        self.model = model
        self.framework = framework
        self.model_type = model_type
        self.metric = 0
        if id:
            self.id = id #set as id if provided
        else:
            self.id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) #otherwise set id
    def set_metric(self, metric):
        self.metric = metric
    def __str__(self):
        return 'MODEL DETAILS: ' + self.model_type + ' model from ' + self.framework
    

def clean_and_scale_dataset(dirty_df_dict, na_action='mean', scaler=None, class_col='class'):
    
    clean_df_list = []

    if scaler:
        scaler.fit(dirty_df_dict['train'].loc[:,dirty_df_dict['train'].columns!=class_col])
    for df_name, dirty_df in dirty_df_dict.items():
        print(f'number of initial rows: {dirty_df.shape[0]}')

        if scaler:
            dirty_df.loc[:,dirty_df.columns!=class_col] = scaler.transform(dirty_df.loc[:,dirty_df.columns!=class_col])

        #how to handle na values in dataset:
        if na_action == 'drop':
            dirty_df = dirty_df.dropna()
            print(f'number of rows after dropping: {dirty_df.shape[0]}')
        if na_action == 'mean':
            dirty_df = dirty_df.fillna(dirty_df.mean())
        if na_action == 'mode':
            dirty_df = dirty_df.fillna(dirty_df.mode())
        if na_action == 'zeros':
            dirty_df = dirty_df.fillna(0)
        else:
            try:
                dirty_df = dirty_df.fillna(int(na_action))
            except:
                dirty_df = dirty_df.fillna(0) 
                print('filled with zeros as a failover')

        print(f'number of rows after cleaning: {dirty_df.shape[0]}')

        cleaned_df = dirty_df
        clean_df_list.append(cleaned_df)
    
    return clean_df_list

def balance(df, class_col='class', balance_method='downsample'):

    if type(balance_method) == int:
        n_samples = balance_method
    elif type(balance_method) == str:
        if balance_method == 'downsample':
            n_samples = min(df[class_col].value_counts())
        if balance_method == 'upsample':
            n_samples = max(df[class_col].value_counts())
        else:
            raise ValueError('no viable sampling method provided, please enter (upsample, downsample, or an integer)')

    df_list = []
    for label in np.unique(df[class_col]):
        subset_df = df[df[class_col]==label]
        resampled_subset_df = resample(subset_df, 
                                        replace=(subset_df.shape[0]<n_samples),    # sample with replacement if less than number of samples, otherwise without replacement
                                        n_samples=n_samples)    # to match minority class
        df_list.append(resampled_subset_df)
    balanced_df = pd.concat(df_list)

    return balanced_df


def prep_data(df_dict, shuffle_data=True, balance_method='downsample'):
    '''
    always pass training set as first df in list
    '''
    #encode dataset to binary variables
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df_dict['train']['class'])

    prepped_df_list = []
    
    for df_key, df in df_dict.items():

        #encode training dataset
        df['class'] = encoder.transform(df['class'])

        if balance_method:
            if df_key=='train': #only balance training data
                df = balance(df, class_col='class', balance_method=balance_method)

        dataset_df = Dataset(df, 'class')
        if shuffle_data:
            dataset_df.data, dataset_df.target = shuffle(dataset_df.data, dataset_df.target)
        
        prepped_df_list.append(dataset_df)

    return prepped_df_list, encoder

def generate_model_id(model_type, metric):
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return str(model_type) + '_' + now_time + '_' +  str(round(metric,4))

def pickle_save_model(algo, model_folder='models'):
    '''
    save the model with datetime if with_datetime=True, which will save model_20190202122930
    '''
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    filename = model_folder+'/'+str(algo.model_type)+'_'+str(algo.id)+'.model'
    print(f'saving as file: {filename}')
    pickle.dump(algo, open(filename, 'wb'))
    return None

def pickle_load_model(model_path=None, models_directory=None, model_type=None, model_selection_criteria='best'):
    '''
    if no model_full_path is provided, then assume that we are looking for the most recent model
    model type can also be specified to retrieve the most recent model of a current type
    '''
    if model_path:
        model_path = model_path
    else:
        model_path = './'
        if models_directory:
            model_path += models_directory + '/'
            if model_type:
                model_path += model_type
        all_possible_files = glob.glob(model_path+'*')
        print([all_possible_files])

        if model_selection_criteria == 'recent':
            #separate out the timestamps
            timestamps = [x.split('_')[-1].strip('.model') for x in all_possible_files]
            #find index of max timestamp
            last_timestamp = max(timestamps)
            most_recent_index = [i for i, j in enumerate(timestamps) if j == last_timestamp][0]
            model_path = all_possible_files[most_recent_index]
            print('most recent model found at:', model_path)
        else:
            model_path = all_possible_files[-1]
            print('last model selected:', model_path)
        

    try:
        model = pickle.load(open(model_path, 'rb'))
        print('model successfully loaded from: {}'.format(model_path))
        return model
    except:
        print('did not successfully load model from: {}'.format(model_path))
        FileNotFoundError('model file not found')