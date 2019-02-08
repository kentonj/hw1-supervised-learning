

import pandas as pd 
import datetime
import time

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, learning_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, auc, accuracy_score, roc_curve, balanced_accuracy_score


from plot_utils import plot_confusion_matrix, plot_learning_curve
from etl_utils import *


#general params:
USE_DATASET = 'spam' #one of ('spam', 'aps')
PRETRAINED_MODELS_DIR = None # one of (None, directory to models), if not None - looks for each model inside the directory
PRETRAINED_MODEL_FILEPATH = None # one of (None, directory to models) if not None - looks for the specified model
RANDOM_STATE = 27
PLOT_ACTION = 'save' # (None, 'save', 'show') # TODO: TURN THIS TO NONE BEFORE TURNING IN TO AVOID MATPLOTLIB ISSUES
N_LC_CHUNKS = 2 #number of chunks for learning curve data segmentation
N_CV = 5 # number of kfold cross validation splits, 1/N_CV computes the validation percentage, if any
N_EPOCHS = 20 # number of epochs for neural network training
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample')
SCORING_METRIC = 'roc_auc' #this works well for both balanced and imbalanced classification problems

def generate_decision_tree(id, n_features=None, n_classes=None):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=5, min_samples_split=0.02)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree', framework='sklearn', id=id)

def generate_svm_rbf(id, n_features=None, n_classes=None):
    svm_classifier = SVC(C=5, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachineRBF', framework='sklearn', id=id)

def generate_svm_poly(id, n_features=None, n_classes=None):
    svm_classifier = SVC(C=5, kernel='poly', degree=3, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachinePoly', framework='sklearn', id=id)

def generate_svm_linear(id, n_features=None, n_classes=None):
    svm_classifier = SVC(C=5, kernel='linear', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachineLinear', framework='sklearn', id=id)

def generate_knn_20(id, n_features=None, n_classes=None):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors20', framework='sklearn', id=id)

def generate_knn_15(id, n_features=None, n_classes=None):
    knn_classifier = KNeighborsClassifier(n_neighbors=15)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors15', framework='sklearn', id=id)

def generate_knn_10(id, n_features=None, n_classes=None):
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors10', framework='sklearn', id=id)

def generate_knn_5(id, n_features=None, n_classes=None):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors5', framework='sklearn', id=id)

def generate_gradient_boosting_trees(id, n_features=None, n_classes=None):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree', framework='sklearn', id=id)

def generate_sk_mlp_classifier(id, n_features=None, n_classes=None):
    gbt_classifier = MLPClassifier(hidden_layer_sizes=(100,20,) ,random_state=RANDOM_STATE, max_iter=N_EPOCHS,)
    return MachineLearningModel(gbt_classifier, model_type='MLPClassifier', framework='sklearn', id=id)


def train_model(algo, training_data, n_folds=5, n_chunks=5):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, runs a learning curve
    and fits the model
    '''
    if algo.framework == 'sklearn':
        print('creating learning curves...'.format(algo.model_type))
        train_sizes, train_scores, val_scores = learning_curve(algo.model, 
                                                                training_data.data, 
                                                                training_data.target, 
                                                                scoring=SCORING_METRIC,
                                                                cv=n_folds, 
                                                                train_sizes=np.linspace(0.1, 1.0, n_chunks))
        
        print('training model...'.format(algo.model_type))
        training_start = time.time()
        algo.model.fit(training_data.data, training_data.target)
        training_end = time.time()
        algo.set_training_time(training_end-training_start)
        print('time to train: {} seconds'.format(algo.training_time))

        cv_scores = cross_val_score(algo.model, training_data.data, training_data.target, scoring=SCORING_METRIC, cv=n_folds)
        algo.cv_score = cv_scores.mean()
        print('average cross validation {} score: {} (+/- {})'.format(SCORING_METRIC, round(cv_scores.mean(),2), round(cv_scores.std() * 2,2)))
        if PLOT_ACTION:
            plot_learning_curve(algo, 
                                train_sizes, 
                                train_scores, 
                                val_scores, 
                                title=('Learning Curve - '+str(algo.model_type)),
                                figure_action=PLOT_ACTION, 
                                figure_path=('figures/'+str(algo.id)),
                                file_name=(str(algo.model_type)+'_LC'))

    return algo


def evaluate_model(algo, test_data, classes_list, figure_action='save'):
    '''
    using the fitted model, 
    '''
    print('evaluating model...')
    if algo.framework == 'sklearn':
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
                                file_name=(str(algo.model_type)+'_CM'))
        plot_confusion_matrix(cm, 
                                algo, 
                                classes=classes_list, 
                                normalize=True, 
                                title=('Normalized Confusion Matrix - '+str(algo.model_type)),
                                figure_action=PLOT_ACTION, 
                                figure_path=('figures/'+str(algo.id)),
                                file_name=(str(algo.model_type)+'_CM_normalized'))

    fp_rate, tp_rate, thresh = roc_curve(test_data.target, predictions)
    print('precision         |', round(precision_score(test_data.target, predictions),2))
    print('recall            |', round(recall_score(test_data.target, predictions),2))
    print('F1                |', round(f1_score(test_data.target, predictions),2))
    print('ROC-AUC           |', round(roc_auc_score(test_data.target, predictions),2))
    print('accuracy          |', round(accuracy_score(test_data.target,predictions),2))
    print('balanced accuracy |', round(balanced_accuracy_score(test_data.target,predictions),2))
    return None


def main():

    algo_batch_id = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) #set ID for one run, so all the algos have the same ID
    algo_generator_dict = {
        'MLPClassifier': generate_sk_mlp_classifier,
        'DecisionTree': generate_decision_tree,
        'SupportVectorMachineRBF': generate_svm_rbf,
        'SupportVectorMachinePoly': generate_svm_poly,
        'SupportVectorMachineLinear': generate_svm_linear,
        'KNearestNeighbors20': generate_knn_20,
        'KNearestNeighbors15': generate_knn_15,
        'KNearestNeighbors10': generate_knn_10,
        'KNearestNeighbors5': generate_knn_5,
        'GradientBoostingTree': generate_gradient_boosting_trees
    }

    #load dataset
    
    if USE_DATASET == 'spam':
        df = pd.read_csv('data/spam/spambasedata.csv', sep=',')
        print('using the dataset stored in ./data/spam')
        #shuffle data before splitting to train and test
        df = df.sample(frac=1).reset_index(drop=True)
        train_frac = 0.8
        train_samples = int(round(df.shape[0]*train_frac))
        dirty_train_df = df.iloc[:train_samples,:]
        dirty_test_df = df.iloc[train_samples:,:]
        class_col = 'class'

    elif USE_DATASET == 'aps':
        dirty_train_df = pd.read_csv('data/aps/aps_failure_training_set.csv', na_values=['na'])
        dirty_test_df = pd.read_csv('data/aps/aps_failure_test_set.csv', na_values=['na'])
        print('using the dataset stored in ./data/aps')
        class_col = 'class'

    #clean both datasets
    scaler = preprocessing.MinMaxScaler()
    [train_df, test_df] = clean_and_scale_dataset({'train':dirty_train_df, 'test':dirty_test_df}, scaler=scaler ,na_action=-1)

    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data({'train':train_df, 'test':test_df}, shuffle_data=True, balance_method=BALANCE_METHOD, class_col=class_col)
    print('\nTRAINING DATA INFORMATION')
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print('size of training dataset:', train_dataset.data.shape)
    print('class counts:\n', train_dataset.df[class_col].value_counts())

    
    # if LOAD_PRETRAINED_MODELS:
    #     for algo_key, algo_generator in algo_generator_dict.items():
    #         try:
    #             algo = pickle_load_model(models_directory='models', model_type=algo_key, model_selection_criteria='recent')
    #             print('loaded algo:\n', algo_key)
    #         except:
    #             print('\ntraining: {} -- loading the specified algorithm failed'.format(algo_key))
    #             algo = algo_generator_dict[algo_key](id=algo_batch_id, n_features=train_dataset.data.shape[1], n_classes=len(label_encoder.classes_))
    #             algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
    #         finally:
    #             raise Exception('failed to load or train algorithm, aborting')
    if PRETRAINED_MODEL_FILEPATH:
        try:
            algo = pickle_load_model(model_path=PRETRAINED_MODEL_FILEPATH)
            print('loaded algo: {} - {}'.format(algo.model_type, algo.id))
            evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
            return None #break out of main function early if a single file is specified
        except:
            raise Exception('failed to load specified model, aborting')
            
    for algo_key, algo_generator in algo_generator_dict.items():
        print('\nModel Name: {}'.format(algo_key))
        if PRETRAINED_MODELS_DIR:
            try:
                algo = pickle_load_model(models_directory=PRETRAINED_MODELS_DIR, model_type=algo_key)
                print('loaded algo: {} - {}'.format(algo.model_type, algo.id))
            except:
                try:
                    print('\ntraining: {} -- loading the specified algorithm failed'.format(algo_key))
                    algo = algo_generator_dict[algo_key](id=algo_batch_id, n_features=train_dataset.data.shape[1], n_classes=len(label_encoder.classes_))
                    algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
                except:
                    raise Exception('failed to load or train algorithm, aborting')
        
        else:
            algo = algo_generator_dict[algo_key](id=algo_batch_id, n_features=train_dataset.data.shape[1], n_classes=len(label_encoder.classes_))
            algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
            pickle_save_model(algo, model_folder=('models/'+str(algo.id)))

        try:        
            evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
        except:
            raise Exception('unable to evaluate model')
    
if __name__ == '__main__': 
    ## general params:
    
    main()
    