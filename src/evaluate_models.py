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
PLOT_ACTION = 'save' # (None, 'save', 'show')
N_LC_CHUNKS = 10 #number of chunks for learning curve data segmentation
N_CV = 10 # number of kfold cross validation splits

def generate_decision_tree(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=8)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree', framework='sklearn', id=id)

def generate_svm_rbf(id):
    svm_classifier = SVC(C=0.7, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachineRBF', framework='sklearn', id=id)

def generate_svm_poly(id):
    svm_classifier = SVC(C=0.7, kernel='poly', degree=3, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachinePoly', framework='sklearn', id=id)

def generate_svm_linear(id):
    svm_classifier = SVC(C=0.7, kernel='linear', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachineLinear', framework='sklearn', id=id)

def generate_knn(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors', framework='sklearn', id=id)

def generate_gradient_boosting_trees(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.15, n_estimators=50, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree', framework='sklearn', id=id)

def generate_neural_network(id):
    nn_classifier = Sequential([
        Dense(32, input_shape=()),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    return MachineLearningModel(nn_classifier, model_type='GradientBoostingTree', framework='sklearn', id=id)



def train_sklearn_model(algo, training_data, n_folds=5, n_chunks=5):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, runs a learning curve
    and fits the model
    '''

    train_sizes, train_scores, val_scores = learning_curve(algo.model, 
                                                            training_data.data, 
                                                            training_data.target, 
                                                            scoring='f1',
                                                            cv=n_folds, 
                                                            train_sizes=np.linspace(0.1, 1.0, n_chunks))

    algo.model.fit(training_data.data, training_data.target)

    cv_scores = cross_val_score(algo.model, training_data.data, training_data.target, scoring='f1', cv=n_folds)
    algo.cv_score = cv_scores.mean()
    print("Cross Validation Average Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    if PLOT_ACTION:
        plot_learning_curve(algo, 
                            train_sizes, 
                            train_scores, 
                            val_scores, 
                            title=('Learning Curve - '+str(algo.model_type)),
                            figure_action=PLOT_ACTION, 
                            figure_path=('figures/'+str(algo.id)),
                            file_name=(str(algo.model_type)+'_'+str(algo.id)+'LC'))

    return algo


def evaluate_model(algo, test_data, classes_list, figure_action='save'):
    '''
    using the fitted model, 
    '''
    predictions = algo.model.predict(test_data.data)
    cm = confusion_matrix(test_data.target, predictions)
    if PLOT_ACTION:
        plot_confusion_matrix(cm, 
                                algo, 
                                classes=classes_list, 
                                normalize=False, 
                                title=('Confusion Matrix - '+str(algo.model_type)),
                                figure_action=PLOT_ACTION, 
                                figure_path=('figures/'+str(algo.id)),
                                file_name=(str(algo.model_type)+'_'+str(algo.id)+'_'+'CM'))
        plot_confusion_matrix(cm, 
                                algo, 
                                classes=classes_list, 
                                normalize=True, 
                                title=('Normalized Confusion Matrix - '+str(algo.model_type)),
                                figure_action=PLOT_ACTION, 
                                figure_path=('figures/'+str(algo.id)),
                                file_name=(str(algo.model_type)+'_'+str(algo.id)+'_'+'CM_normalized'))

    final_score = f1_score(test_data.target, predictions)
    algo.test_score = final_score
    print('final f1-score on test data', algo.test_score)
    return algo.test_score


def main():

    algo_batch_id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    algo_generator_dict = {
        'DecisionTree': generate_decision_tree,
        'SupportVectorMachineRBF': generate_svm_rbf,
        'SupportVectorMachinePoly': generate_svm_poly,
        'SupportVectorMachineLinear': generate_svm_linear,
        'KNearestNeighbors': generate_knn,
        'GradientBoostingTrees': generate_gradient_boosting_trees
    }

    #load dataset
    dirty_train_df = pd.read_csv('data/aps_failure_training_set.csv', na_values=['na'])
    dirty_test_df = pd.read_csv('data/aps_failure_test_set.csv', na_values=['na'])

    #clean both datasets
    scaler = preprocessing.MinMaxScaler()
    [train_df, test_df] = clean_and_scale_dataset({'train':dirty_train_df, 'test':dirty_test_df}, scaler=scaler ,na_action=-1)

    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data({'train':train_df, 'test':test_df}, shuffle_data=True, balance_method=1000)
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print('size of training dataset:', train_dataset.data.shape)
    print('size of testing dataset:', test_dataset.data.shape)

    
    if LOAD_PRETRAINED_MODELS:
        for algo_key in algo_generator_dict.keys():
            algo = pickle_load_model(models_directory='models', model_type=algo_key, model_selection_criteria='recent')
            print('algo:\n', algo)
            model_score = evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
    else:
        for algo_key, algo_generator in algo_generator_dict.items():
            print('\ntraining: {}'.format(algo_key))
            algo = algo_generator_dict[algo_key](id=algo_batch_id)
            if algo.framework == 'sklearn':
                algo = train_sklearn_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
            else:
                print(f'nothing yet to implement a {algo.framework} model')
                algo = None

            if algo:
                pickle_save_model(algo)

            model_score = evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
    
    
if __name__ == '__main__': 
    ## general params:
    
    main()
    