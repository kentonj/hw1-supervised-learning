

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
N_LC_CHUNKS = 10 #number of chunks for learning curve data segmentation
N_CV = 5 # number of kfold cross validation splits, 1/N_CV computes the validation percentage, if any
N_EPOCHS = 2000 # maximum number of epochs for neural network training
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample')
SCORING_METRIC = 'roc_auc' #this works well for both balanced and imbalanced classification problems

def generate_decision_tree_more_pruning(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=3, min_samples_split=0.08)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree-MorePruning', framework='sklearn', id=id)

def generate_decision_tree_less_pruning(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=5, min_samples_split=0.02)
    return MachineLearningModel(dt_classifier, model_type='DecisionTree-LessPruning', framework='sklearn', id=id)

def generate_svm_rbf_c1(id):
    svm_classifier = SVC(C=1, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-RBF-C1', framework='sklearn', id=id)

def generate_svm_rbf_c5(id):
    svm_classifier = SVC(C=5, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-RBF-C5', framework='sklearn', id=id)
    
def generate_svm_rbf_c10(id):
    svm_classifier = SVC(C=10, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-RBF-C10', framework='sklearn', id=id)

def generate_svm_poly_c5_degree2(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=2, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Poly-C5-Degree2', framework='sklearn', id=id)

def generate_svm_poly_c5_degree3(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=3, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Poly-C5-Degree3', framework='sklearn', id=id)

def generate_svm_poly_c5_degree4(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=4, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Poly-C5-Degree4', framework='sklearn', id=id)

def generate_svm_linear_c1(id):
    svm_classifier = SVC(C=1, kernel='linear', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Linear-C1', framework='sklearn', id=id)

def generate_svm_linear_c5(id):
    svm_classifier = SVC(C=5, kernel='linear', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Linear-C5', framework='sklearn', id=id)

def generate_svm_linear_c10(id):
    svm_classifier = SVC(C=10, kernel='linear', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_type='SupportVectorMachine-Linear-C10', framework='sklearn', id=id)

def generate_knn_20_euclidean(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors-K20-DistanceEuclidean', framework='sklearn', id=id)

def generate_knn_15_euclidean(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=15, p=2)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors-K15-DistanceEuclidean', framework='sklearn', id=id)

def generate_knn_10_manhattan(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=10, p=1)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors-K10-DistanceManhattan', framework='sklearn', id=id)

def generate_knn_10_euclidean(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=10, p=2)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors-K10-DistanceEuclidean', framework='sklearn', id=id)

def generate_knn_5_euclidean(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    return MachineLearningModel(knn_classifier, model_type='KNearestNeighbors-K5-DistanceEuclidean', framework='sklearn', id=id)

def generate_gradient_boosting_trees_lr01(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree-LearningRate0.01', framework='sklearn', id=id)

def generate_gradient_boosting_trees_lr05(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree-LearningRate0.05', framework='sklearn', id=id)

def generate_gradient_boosting_trees_lr1(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree-LearningRate0.1', framework='sklearn', id=id)

def generate_gradient_boosting_trees_lr5(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.5, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_type='GradientBoostingTree-LearningRate0.5', framework='sklearn', id=id)

def generate_sk_mlp_classifier_lr001(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.001)
    return MachineLearningModel(mlp_classifier, model_type='MLPClassifier-LearningRate0.001', framework='sklearn', nn=True, id=id)

def generate_sk_mlp_classifier_lr01(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.01)
    return MachineLearningModel(mlp_classifier, model_type='MLPClassifier-LearningRate0.01', framework='sklearn', nn=True, id=id)

def generate_sk_mlp_classifier_lr1(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.1)
    return MachineLearningModel(mlp_classifier, model_type='MLPClassifier-LearningRate0.1', framework='sklearn', nn=True, id=id)

def train_model(algo, training_data, n_folds=5, n_chunks=5):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, runs a learning curve
    and fits the model
    '''
    if algo.framework == 'sklearn':
        
        if algo.nn == False:
            print('creating learning curves...')
            train_sizes, train_scores, val_scores = learning_curve(algo.model, 
                                                                    training_data.data, 
                                                                    training_data.target, 
                                                                    scoring=SCORING_METRIC,
                                                                    cv=n_folds, 
                                                                    train_sizes=np.linspace(0.1, 1.0, n_chunks))
            # set train sizes as the number of epochs

        print('training model...')
        training_start = time.time()
        algo.model.fit(training_data.data, training_data.target)
        training_end = time.time()
        algo.set_training_time(training_end-training_start)
        print('time to train: {} seconds'.format(algo.training_time))

        #neural network training curves are based on epochs
        if algo.nn == True:
            train_scores = algo.model.loss_curve_ 
            val_scores = algo.model.validation_scores_
            train_sizes = list([i for i in range(algo.model.n_iter_)])

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
    eval_start = time.time()
    predictions = algo.model.predict(test_data.data)
    eval_end = time.time()
    algo.set_evaluation_time(eval_end-eval_start)
    print('time to predict: {} seconds'.format(algo.evaluation_time))
    
    
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
        'DecisionTree-MorePruning': generate_decision_tree_more_pruning,
        'DecisionTree-LessPruning': generate_decision_tree_less_pruning,
        'SupportVectorMachine-RBF-C1': generate_svm_rbf_c1,
        'SupportVectorMachine-RBF-C5': generate_svm_rbf_c5,
        'SupportVectorMachine-RBF-C10': generate_svm_rbf_c10,
        'SupportVectorMachine-Poly-C5-Degree2': generate_svm_poly_c5_degree2,
        'SupportVectorMachine-Poly-C5-Degree3': generate_svm_poly_c5_degree3,
        'SupportVectorMachine-Poly-C5-Degree4': generate_svm_poly_c5_degree4,
        'SupportVectorMachine-Linear-C1': generate_svm_linear_c1,
        'SupportVectorMachine-Linear-C5': generate_svm_linear_c5,
        'SupportVectorMachine-Linear-C10': generate_svm_linear_c10,
        'KNearestNeighbors-K20-DistanceEuclidean': generate_knn_20_euclidean,
        'KNearestNeighbors-K15-DistanceEuclidean': generate_knn_15_euclidean,
        'KNearestNeighbors-K10-DistanceManhattan': generate_knn_10_manhattan,
        'KNearestNeighbors-K10-DistanceEuclidean': generate_knn_10_euclidean,
        'KNearestNeighbors-K5-DistanceEuclidean':generate_knn_5_euclidean,
        'GradientBoostingTree-LearningRate0.01': generate_gradient_boosting_trees_lr01,
        'GradientBoostingTree-LearningRate0.05': generate_gradient_boosting_trees_lr05,
        'GradientBoostingTree-LearningRate0.1': generate_gradient_boosting_trees_lr1,
        'GradientBoostingTree-LearningRate0.5': generate_gradient_boosting_trees_lr5,
        'MLPClassifier-LearningRate0.001': generate_sk_mlp_classifier_lr001,
        'MLPClassifier-LearningRate0.01': generate_sk_mlp_classifier_lr01,
        'MLPClassifier-LearningRate0.1': generate_sk_mlp_classifier_lr1
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
        ## TODO: consider cleaning this up to look in a directory then pull all models in that directory, iterating through and evaluating each one
        if PRETRAINED_MODELS_DIR:
            try:
                algo = pickle_load_model(models_directory=PRETRAINED_MODELS_DIR, model_type=algo_key)
                print('loaded algo: {} - {}'.format(algo.model_type, algo.id))
            except:
                try:
                    print('\ntraining: {} -- loading the specified algorithm failed'.format(algo_key))
                    algo = algo_generator_dict[algo_key](id=algo_batch_id)
                    algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
                except:
                    raise Exception('failed to load or train algorithm, aborting')
        
        else:

            algo = algo_generator_dict[algo_key](id=algo_batch_id)
            algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
            pickle_save_model(algo, model_folder=('models/'+str(algo.id)))

        try:        
            evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
        except:
            raise Exception('unable to evaluate model')
    
if __name__ == '__main__': 
    ## general params:
    
    main()
    