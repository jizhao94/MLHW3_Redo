'''
Machine Learning Pipeline for Homework 3
'''

import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from datetime import datetime
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
'''
Import the data in csv file
'''
def read_data(filename):
    '''
    Convert csv file to a pandas dataframe
    Input: filename (str)
    '''
    data = pd.read_csv(filename)

    return data

'''
Convert datatypes
'''
def convert_type(data, colname, target_type):
    '''
    Convert datatype of a column
    Input: data (dataframe), colname (str), type (str)
    '''
    data[colname] = data[colname].astype(target_type)

'''
Check if the column contains NULLs
'''
def if_null(data, colname):
    '''
    Input: data(dataframe)，colname(str)
    '''
    return data[colname].isnull().values.any()

'''
Find the distribution of a variable in the dataset
'''
def describe_data(data, colname):
    '''
    Input: data(dataframe), colname(str)
    '''
    return data[colname].describe()

'''
Make the boxplot of a variable in the dataset
'''
def boxplot(data, colname):
    '''
    Input: data(dataframe), colname(str)
    '''
    return data[colname].plot.box()

'''
Make the density plot of a variable in the dataset
'''
def density_plot(data, colname):
    '''
    Input: data(dataframe), colname(str)
    '''
    return data[colname].plot.density()

'''
Find summaries of all variables that we are interested in
'''
def find_summaries(data, colnames):
    '''
    Input: data(dataframe), colnames (list)
    '''
    return data[colnames].describe()

'''
Find correlations between variables
'''
def find_corr(data, col1, col2):
    '''
    Input: data(dataframe). col1(str), col2(str)
    '''
    return data[col1].corr(data[col2])

'''
Discretize a set of columns in a dataset
'''
def discretize_col(data, columns):
    '''
    To discretize the continuous variable into three discrete variables: 0, 1, and 2;
    the boundaries are the minimum value, the 25% quantile, the 75% quantile, and the maximum value.
    
    Inputs: data, pandas dataframe
            columns, list
    '''
    for column in columns:
        data[column] = pd.cut(data[column], bins=[data[column].min(), data[column].quantile(0.25), data[column].quantile(0.75),
                                                data[column].max()], labels=[0,1,2], include_lowest=True)
'''
Fill in NA values with mean
'''
def fill_na(data, columns):
    '''
    Input: data, pandas dataframe
           columns, list
    '''
    for column in columns:
        if data[column].isnull().any():
            data[column] = data[column].fillna(data[column].median())

'''
Convert the label to dummy variables
'''
def label_to_dummy(item, bar):
    '''
    item: int
    bar: int
    '''
    if item >= bar:
        result = 1
    else:
        result = 0
    return result

def cat_to_dummy(item, bar):
    '''
    item: str
    bar: str
    '''
    if item == bar:
        result = 1
    else:
        result = 0
    return result

'''
Convert columns in categorical variables to dummy variables
'''
def to_dummy(data, column):
    '''
    data, pandas dataframe
    column, list
    '''
    data = pd.get_dummies(data, columns=column)
    
    return data

'''
Slice time series data by time
'''
def slice_time_data(data, column, start_time, end_time):
    '''
    data, pandas dataframe
    column, str
    start_time, string
    end_time, string
    '''
    return data[(data[column] >= start_time) & (data[column] <= end_time)]


'''
The following codes are referenced from the folloiwng website:
https://github.com/rayidghani/magicloops/blob/master/simpleloop.py, credit to Rayid Ghani
'''
def define_clfs_params():
    
    clfs = {
        'BG': BaggingClassifier(n_estimators=10),
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.LinearSVC(random_state=0, penalty='l1', dual=False),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'NB': GaussianNB()}

    grid = {
        'BG': {'n_estimators': [10,100]}, 
        'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2']},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10]},
        'GB': {'n_estimators': [1,10,100], 'max_features': [3,5,10]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10]},
        'SVM' :{'penalty':['l1','l2'], 'C' :[0.01,0.1,1]},
        'KNN' :{'n_neighbors': [1,5,10,25],'weights': ['uniform','distance']},
        'NB': {}
        }
    
    return clfs, grid
    


def joint_sort_descending(l1, l2):
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)
    return recall

def f1_at_k(y_true, y_scores, k):

    precision = precision_at_k(y_true, y_scores, k)
    recall = recall_at_k(y_true, y_scores, k)

    return 2 * (precision * recall)/(precision + recall)


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.title(model_name)
    plt.show()


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df = pd.DataFrame(columns=('model_type', 'clf', 'parameters','auc-roc','p_at_5',
                                        'p_at_10', 'p_at_20', 'r_at_5', 'r_at_10', 'r_at_20', 'f1_at_5',
                                        'f1_at_10', 'f1_at_20'))
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            clf.set_params(**p)
            if 'SVM' in models_to_run[index]:
                y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
            else:
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
            y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
            results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                              roc_auc_score(y_test, y_pred_probs),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                              precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                              recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                              f1_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)]
            plot_precision_recall_n(y_test,y_pred_probs,clf)
    return results_df


