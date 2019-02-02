import pandas as pd 
import datetime
import pickle
import os
import glob

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
    def __init__(self, model, model_type, framework):
        self.model = model
        self.framework = framework
        self.model_type = model_type
        self.metric = 0
    def set_metric(self, metric):
        self.metric = metric
    def __str__(self):
        return 'this is a ' + model_type + ' model from ' + framework
    

def clean_dataset(dirty_df_list, na_action='mean'):
    
    clean_df_list = []

    for dirty_df in list(dirty_df_list):
        print(f'number of initial rows: {dirty_df.shape[0]}')

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


def prep_data(training_and_test_df_list, scale=False):
    '''
    always pass training set as first df in list
    '''
    #encode dataset to binary variables
    prepped_df_list = []
    for df_index in range(len(list(training_and_test_df_list))):
        df = training_and_test_df_list[df_index]
        if df_index == 0:
            encoder = preprocessing.LabelEncoder() #outside of function scope so that it can be referenced again
            encoder.fit(df['class'])

        #encode training dataset
        df['class'] = encoder.transform(df['class'])
        prepped_df_list.append(Dataset(df,'class'))

    return prepped_df_list, encoder

def generate_model_id(model_type, metric):
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return str(model_type) + '_' + now_time + '_' +  str(round(metric,4))

def pickle_save_model(model, model_id, model_folder='models'):
    '''
    save the model with datetime if with_datetime=True, which will save model_20190202122930
    '''
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    filename = model_folder+'/'+model_id+'.model'
    print(f'saving as file: {filename}')
    pickle.dump(model, open(filename, 'wb'))
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

        if model_selection_criteria == 'best':
            #separate out metrics
            metrics = [float(x.split('_')[-1].strip('.model')) for x in all_possible_files]
            #find index of max timestamp
            best_metric = max(metrics)
            best_metric_index = [i for i, j in enumerate(metrics) if j == best_metric][0] #default takes the first item in the list if multiple are returned
            model_path = all_possible_files[best_metric_index]
            print('best metric model found at:', model_path, 'with: ', best_metric)

        elif model_selection_criteria == 'recent':
            #separate out the timestamps
            timestamps = [x.split('_')[-2].strip('.model') for x in all_possible_files]
            #find index of max timestamp
            last_timestamp = max(timestamps)
            most_recent_index = [i for i, j in enumerate(timestamps) if j == last_timestamp][0]
            model_path = all_possible_files[most_recent_index]
            print('most recent model found at:', model_path)

    try:
        model = pickle.load(open(model_path, 'rb'))
        print('model successfully loaded from: {}'.format(model_path))
        model_id = model_path.split('/')[-1].strip('.model')
        print('model id:', model_id)
        return model, model_id
    except:
        print('did not successfully load model from: {}'.format(model_path))
        FileNotFoundError('model file not found')