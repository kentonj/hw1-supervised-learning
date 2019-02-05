import platform
import matplotlib as mpl
if platform.mac_ver()[0] != '':
    print('mac os version detected:', platform.mac_ver()[0], ' - switching matplotlib backend to TkAgg')
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import os
import itertools


def plot_confusion_matrix(cm, algo, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figure_action='show', figure_path='figures/cm', file_name=None):
    '''
    heavily adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    '''
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_type)+'_'+str(algo.id)+'.png')
    plt.close()
    return None



def plot_learning_curve(algo, train_sizes, train_scores, val_scores, title='Learning Curve', figure_action='show', figure_path='figures/lc', file_name=None):
    '''
    heavily adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    '''
    plt.figure()
    plt.title(title)
    
    plt.ylabel("Score")

    if algo.framework == 'sklearn':
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        plt.xlabel("Training examples")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='orange')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='orange',
             label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color='blue',
                label="Cross-validation score")
    elif algo.framework == 'keras':
        #if there is only one value, not a range of values for k-fold cross validation
        plt.xlabel("Epochs")
        plt.plot(train_sizes, train_scores, 'o-', color='orange',
             label="Training score")
        plt.plot(train_sizes, val_scores, 'o-', color='blue',
                label="Cross-validation score")
    plt.grid()

    plt.legend(loc="best")
    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_type)+'_'+str(algo.id)+'.png')
    plt.close()
    return None

