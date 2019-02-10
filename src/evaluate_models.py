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


from plot_utils import plot_confusion_matrix, plot_model_family_learning_curves
from etl_utils import *


#general params:
USE_DATASET = 'aps' #one of ('spam', 'aps')
SAVE_MODELS = False #set to true if you want to pickle save your model
PRETRAINED_MODEL_FILEPATH = None # one of (None, directory to models) - default to None to train models, if not None - looks for the specified model
RANDOM_STATE = 27
PLOT_ACTION = None # (None, 'save', 'show') - default to None to avoid issues with matplotlib depending on OS
N_LC_CHUNKS = 10 #number of chunks for learning curve data segmentation
N_CV = 5 # number of kfold cross validation splits, 1/N_CV computes the validation percentage, if any
N_EPOCHS = 2000 # maximum number of epochs for neural network training
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample')
SCORING_METRIC = 'roc_auc' #this works well for both balanced and imbalanced classification problems

def generate_decision_tree_most_pruning(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=2, min_samples_split=0.2)
    return MachineLearningModel(dt_classifier, model_family='DecisionTree', model_type='DT-MostPruning', framework='sklearn', id=id)

def generate_decision_tree_middle_pruning(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=3, min_samples_split=0.08)
    return MachineLearningModel(dt_classifier, model_family='DecisionTree', model_type='DT-MiddlePruning', framework='sklearn', id=id)

def generate_decision_tree_least_pruning(id):
    dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, max_depth=5, min_samples_split=0.02)
    return MachineLearningModel(dt_classifier, model_family='DecisionTree', model_type='DT-LeastPruning', framework='sklearn', id=id)

def generate_svm_rbf_c1(id):
    svm_classifier = SVC(C=1, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-RBF', model_type='SVM-RBF-C1', framework='sklearn', id=id)

def generate_svm_rbf_c5(id):
    svm_classifier = SVC(C=5, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-RBF',model_type='SVM-RBF-C5', framework='sklearn', id=id)
    
def generate_svm_rbf_c100(id):
    svm_classifier = SVC(C=100, kernel='rbf', random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-RBF', model_type='SVM-RBF-C100', framework='sklearn', id=id)

def generate_svm_poly_c5_degree2(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=2, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-Poly', model_type='SVM-Poly-C5-D2', framework='sklearn', id=id)

def generate_svm_poly_c5_degree3(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=3, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-Poly', model_type='SVM-Poly-C5-D3', framework='sklearn', id=id)

def generate_svm_poly_c5_degree4(id):
    svm_classifier = SVC(C=5, kernel='poly', degree=4, random_state=RANDOM_STATE, gamma='scale')
    return MachineLearningModel(svm_classifier, model_family='SVM-Poly', model_type='SVM-Poly-C5-D4', framework='sklearn', id=id)

def generate_knn_20_p2(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    return MachineLearningModel(knn_classifier, model_family='KNN', model_type='KNN-K20-P2', framework='sklearn', id=id)

def generate_knn_15_p2(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=15, p=2)
    return MachineLearningModel(knn_classifier, model_family='KNN', model_type='KNN-K15-P2', framework='sklearn', id=id)

def generate_knn_10_p1(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=10, p=1)
    return MachineLearningModel(knn_classifier, model_family='KNN', model_type='KNN-K10-P1', framework='sklearn', id=id)

def generate_knn_10_p2(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=10, p=2)
    return MachineLearningModel(knn_classifier, model_family='KNN', model_type='KNN-K10-P2', framework='sklearn', id=id)

def generate_knn_5_p2(id):
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    return MachineLearningModel(knn_classifier, model_family='KNN', model_type='KNN-K5-P2', framework='sklearn', id=id)

def generate_gradient_boosting_tree_lr01(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_family='GradientBoostingTree', model_type='GBT-LR.01', framework='sklearn', id=id)

def generate_gradient_boosting_tree_lr05(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_family='GradientBoostingTree', model_type='GBT-LR.05', framework='sklearn', id=id)

def generate_gradient_boosting_tree_lr1(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_family='GradientBoostingTree', model_type='GBT-LR.1', framework='sklearn', id=id)

def generate_gradient_boosting_tree_lr5(id):
    gbt_classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.5, n_estimators=50, max_depth=5, min_samples_split=0.02, random_state=RANDOM_STATE)
    return MachineLearningModel(gbt_classifier, model_family='GradientBoostingTree', model_type='GBT-LR.5', framework='sklearn', id=id)

def generate_sk_mlp_classifier_lr001(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.001)
    return MachineLearningModel(mlp_classifier, model_family='NeuralNetwork', model_type='MLP-LR.001', framework='sklearn', nn=True, id=id)

def generate_sk_mlp_classifier_lr01(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.01)
    return MachineLearningModel(mlp_classifier, model_family='NeuralNetwork', model_type='MLP-LR.01', framework='sklearn', nn=True, id=id)

def generate_sk_mlp_classifier_lr1(id):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,20,), early_stopping=True, n_iter_no_change=50, tol=0.0001, random_state=RANDOM_STATE, max_iter=N_EPOCHS, learning_rate_init=0.1)
    return MachineLearningModel(mlp_classifier, model_family='NeuralNetwork', model_type='MLP-LR.1', framework='sklearn', nn=True, id=id)

def train_model(algo, training_data, n_folds=5, n_chunks=5):
    '''
    use this for a nice wrapper on training and evaluating a ML model from sklearn, runs a learning curve
    and fits the model
    '''
    if algo.nn == False:
        print('generating learning curve data...')
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

    #neural network training curves are based on epochs, not number of samples
    if algo.nn == True:
        train_scores = algo.model.loss_curve_ 
        val_scores = algo.model.validation_scores_
        train_sizes = list([i for i in range(algo.model.n_iter_)])

    cv_scores = cross_val_score(algo.model, training_data.data, training_data.target, scoring=SCORING_METRIC, cv=n_folds)
    algo.cv_score = cv_scores.mean()
    algo.train_sizes = train_sizes
    algo.train_scores = train_scores
    algo.val_scores = val_scores
    print('average cross validation {} score: {} (+/- {})'.format(SCORING_METRIC, round(cv_scores.mean(),2), round(cv_scores.std() * 2,2)))

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
    print('time to predict {} samples: {} seconds'.format(test_data.data.shape[0],algo.evaluation_time))
    
    
    cm = confusion_matrix(test_data.target, predictions)
    algo.cm = cm
    print('confusion matrix:\n',algo.cm)

    fp_rate, tp_rate, thresh = roc_curve(test_data.target, predictions)
    algo.precision = precision_score(test_data.target, predictions)
    algo.recall = recall_score(test_data.target, predictions)
    algo.f1 = f1_score(test_data.target, predictions)
    algo.roc_auc = roc_auc_score(test_data.target, predictions)
    algo.accuracy = accuracy_score(test_data.target,predictions)
    algo.balanced_accuracy = balanced_accuracy_score(test_data.target,predictions)
    print('precision         |', round(algo.precision,2))
    print('recall            |', round(algo.recall,2))
    print('F1                |', round(algo.f1,2))
    print('ROC-AUC           |', round(algo.roc_auc,2))
    print('accuracy          |', round(algo.accuracy,2))
    print('balanced accuracy |', round(algo.balanced_accuracy,2))
    return algo


def main():

    algo_batch_id = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) #set ID for one run, so all the algos have the same ID
    algo_family_generator_dict = {
        'DecisionTree':
            {'DT-MostPruning': generate_decision_tree_most_pruning,
            'DT-MiddlePruning': generate_decision_tree_middle_pruning,
            'DT-LeastPruning': generate_decision_tree_least_pruning},
        'SVM-RBF':
            {'SVM-RBF-C1': generate_svm_rbf_c1,
            'SVM-RBF-C5': generate_svm_rbf_c5,
            'SVM-RBF-C100': generate_svm_rbf_c100},
        'SVM-Poly':
            {'SVM-Poly-C5-D2': generate_svm_poly_c5_degree2,
            'SVM-Poly-C5-D3': generate_svm_poly_c5_degree3,
            'SVM-Poly-C5-D4': generate_svm_poly_c5_degree4},
        'KNN':
            {'KNN-K10-P1': generate_knn_10_p1,
            'KNN-K10-P2': generate_knn_10_p2,
            'KNN-K5-P2': generate_knn_5_p2},
        'GradientBoostingTree':
            {'GBT-LR.01': generate_gradient_boosting_tree_lr01,
            'GBT-LR.05': generate_gradient_boosting_tree_lr05,
            'GBT-LR.1': generate_gradient_boosting_tree_lr1},
        'NeuralNetwork':
            {'MLP-LR.001': generate_sk_mlp_classifier_lr001,
            'MLP-LR.01': generate_sk_mlp_classifier_lr01,
            'MLP-LR.1': generate_sk_mlp_classifier_lr1}
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

    detail_df = pd.DataFrame(columns=['Model Name', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Accuracy', 'Balanced Accuracy', 'Training Time'])
    for algo_family, algo_generator_dict in algo_family_generator_dict.items():
        print('\n\nalgorithm family:', algo_family)
        print('algorithms to test:', [x for x in algo_generator_dict.keys()])
        algo_list = []
        
        for algo_key, algo_generator in algo_generator_dict.items():
            print('\nmodel name: {}'.format(algo_key))
            algo = algo_generator_dict[algo_key](id=algo_batch_id)
            algo = train_model(algo, train_dataset, n_folds=N_CV, n_chunks=N_LC_CHUNKS)
            if SAVE_MODELS:
                pickle_save_model(algo, model_folder='output/'+str(algo.id)+'/models')

            try:        
                evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
            except:
                raise Exception('unable to evaluate model')

            algo_list.append(algo)
            #store algo details in dataframe
            detail_df = detail_df.append({'Model Name': algo.model_type, 
                                        'Precision': algo.precision, 
                                        'Recall':algo.recall, 
                                        'F1':algo.f1, 
                                        'ROC-AUC':algo.roc_auc, 
                                        'Accuracy':algo.accuracy, 
                                        'Balanced Accuracy':algo.balanced_accuracy, 
                                        'Training Time':algo.training_time}, ignore_index=True)

        if PLOT_ACTION:
            plot_model_family_learning_curves(algo_family, 
                                                algo_list, 
                                                figure_action=PLOT_ACTION, 
                                                figure_path='output/'+str(algo_batch_id)+'/figures/lc',
                                                file_name=(str(algo_family)))
            plot_confusion_matrix(algo_family, 
                                    algo_list, 
                                    label_encoder.classes_, 
                                    figure_action=PLOT_ACTION, 
                                    figure_path='output/'+str(algo_batch_id)+'/figures/cm',
                                    file_name=(str(algo_family)))

    # detail_df.to_csv('output/'+str(algo_batch_id)+'/models_summary_'+str(USE_DATASET)+'.csv', sep=',', encoding='utf-8', index=False)
    
if __name__ == '__main__': 
    main()
    