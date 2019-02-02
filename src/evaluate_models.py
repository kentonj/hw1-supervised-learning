import pandas as pd 
import datetime
import pickle
import os
import glob

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from plot_utils import plot_confusion_matrix
from etl_utils import *


#general params:
LOAD_PRETRAINED_MODELS = False
RANDOM_STATE = 27

def generate_decision_tree():
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=8)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree', framework='sklearn')


def train_sklearn_model(training_data, algo):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, returns the model and the cv_scores_mean
    '''
    #define params for each algorithm
    print(f'... Training {algo} ...')
    trained_model = algo.fit(training_data.data, training_data.target)
    cv_scores = cross_val_score(algo, training_data.data, training_data.target, scoring='f1', cv=5)
    print("Cross Validation Average Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    return trained_model, cv_scores.mean()



def evaluate_model(model, model_id, test_data, classes_list, figure_action='save'):
    predictions = model.predict(test_data.data)
    cm = confusion_matrix(test_data.target, predictions)
    plot_confusion_matrix(cm, model_id=model_id, classes=classes_list, normalize=True, figure_action=figure_action)
    score = f1_score(test_data.target, predictions)
    return score


def main():

    algo_generator_dict = {
        'DecisionTree': generate_decision_tree
    }

    #load dataset
    dirty_train_df = pd.read_csv('data/aps_failure_training_set.csv', na_values=['na'])
    dirty_test_df = pd.read_csv('data/aps_failure_test_set.csv', na_values=['na'])

    #clean both datasets
    [train_df, test_df] = clean_dataset([dirty_train_df, dirty_test_df], na_action=-1)

    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data([train_df, test_df])
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    
    if LOAD_PRETRAINED_MODELS:
        for algo_key in algo_generator_dict.keys():
            model, model_id = pickle_load_model(models_directory='models', model_type=algo_key, model_selection_criteria='best')
            print('model:\n', model)
    else:
        for algo_key, algo_generator in algo_generator_dict.items():
            print('training:\n{}'.format(algo_key))
            algo = algo_generator_dict[algo_key]()
            if algo.framework == 'sklearn':
                model, cv_score = train_sklearn_model(train_dataset, algo.model)
                model_id = generate_model_id(algo.model_type, cv_score)
            else:
                print(f'nothing yet to implement a {algo.framework} model')
                model = None

            if model:
                pickle_save_model(model, model_id)
    model_score = evaluate_model(model, model_id, test_dataset, classes_list=label_encoder.classes_)
    
    
if __name__ == '__main__': 
    ## general params:
    
    main()
    