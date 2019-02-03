import pandas as pd 
import datetime
import pickle
import os
import glob

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

from keras.models import Sequential
from keras.layers import Dense, Activation

from plot_utils import plot_confusion_matrix, plot_learning_curve
from etl_utils import *


#general params:
LOAD_PRETRAINED_MODELS = False
RANDOM_STATE = 27

def generate_decision_tree():
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=8)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree', framework='sklearn')

def generate_svm():
    svm_classifier = SVC(random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine', framework='sklearn')

def generate_knn():
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors', framework='sklearn')

def generate_gradient_boosting_trees():
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.15, n_estimators=200, random_state=RANDOM_STATE)

def generate_neural_network():
    nn_classifier = Sequential([
    Dense(32, input_shape=()),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


def train_sklearn_model(algo, training_data, n_folds=5, n_chunks=5):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, runs a learning curve
    and fits the model
    '''

    #use the untrained algo.model to generate a learning curve

    train_sizes, train_scores, val_scores = learning_curve(algo.model, 
                                                            training_data.data, 
                                                            training_data.target, 
                                                            cv=n_folds, 
                                                            train_sizes=np.linspace(0.1, 1.0, n_chunks))

    algo.model.fit(training_data.data, training_data.target)

    cv_scores = cross_val_score(algo.model, training_data.data, training_data.target, scoring='f1', cv=n_folds)
    algo.cv_score = cv_scores.mean()
    print("Cross Validation Average Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

    plot_learning_curve(algo, train_sizes, train_scores, val_scores, figure_action='save')

    return algo


def evaluate_model(algo, test_data, classes_list, figure_action='save'):
    '''
    using the fitted model, 
    '''
    predictions = algo.model.predict(test_data.data)
    cm = confusion_matrix(test_data.target, predictions)
    plot_confusion_matrix(cm, algo, classes=classes_list, normalize=True, figure_action=figure_action)
    final_score = f1_score(test_data.target, predictions)
    algo.test_score = final_score
    print('final f1-score on test data', algo.test_score)
    return algo.test_score


def main():

    algo_generator_dict = {
        'DecisionTree': generate_decision_tree,
        'SupportVectorMachine': generate_svm,
        'KNearestNeighbors': generate_knn,
        'GradientBoostingTrees': generate_gradient_boosting_trees
    }

    #load dataset
    dirty_train_df = pd.read_csv('data/aps_failure_training_set.csv', na_values=['na'])
    dirty_test_df = pd.read_csv('data/aps_failure_test_set.csv', na_values=['na'])

    #clean both datasets
    [train_df, test_df] = clean_dataset([dirty_train_df, dirty_test_df], na_action=-1)

    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data([train_df, test_df], scale_data=True, shuffle_data=True)
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    
    if LOAD_PRETRAINED_MODELS:
        for algo_key in algo_generator_dict.keys():
            algo = pickle_load_model(models_directory='models', model_type=algo_key, model_selection_criteria='recent')
            print('algo:\n', algo)
    else:
        for algo_key, algo_generator in algo_generator_dict.items():
            print('training:\n{}'.format(algo_key))
            algo = algo_generator_dict[algo_key]()
            if algo.framework == 'sklearn':
                algo = train_sklearn_model(algo, train_dataset)
            else:
                print(f'nothing yet to implement a {algo.framework} model')
                algo = None

            if algo:
                pickle_save_model(algo)

    model_score = evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
    
    
if __name__ == '__main__': 
    ## general params:
    
    main()
    