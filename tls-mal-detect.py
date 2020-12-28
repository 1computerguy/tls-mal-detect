#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from features import calculate_pca, random_forest

def svm_analysis(data, label, mal_percent=100.0, model_type='svm', feature_reduction='pca', iterations=1, graph=None, d_format='digit'):
    '''
    Use a Support Vector Machine to classify malware and benign TLS traffic based on metadata
    gathered during the client/server handshake.
    '''
    # Get a percentage of the malware data set to imbalance measurements for analysis
    malware = data[data.malware_label == 1]
    benign = data[data.malware_label == 0].reset_index(drop=True)
    percent = int(len(malware) * (mal_percent / 100))
    malware = malware.sample(n=percent).reset_index(drop=True)
    #print("Malware count: {}".format(malware.shape))

    data = benign.append(malware).reset_index(drop=True)

    # Select feature reduction portion
    if feature_reduction == 'pca':
        feature_data = calculate_pca(data, 'malware_label')
    elif feature_reduction == 'forest':
        feature_data = random_forest(data, 'malware_label', 1000)

    score_list = []
    # Iterate over data and generate list of F1 scores to return for analysis (defualt is only on iteration)
    for round in range(0, iterations):
        X = feature_data.drop([label], axis=1)
        y = feature_data[label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Select type of SVM to use, standard Support vector or One-class SVM
        if model_type == 'svm':
            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train)
            predict = svclassifier.predict(X_test)
            score = f1_score(y_test, predict, pos_label=1)

        elif model_type == 'one':
            nu_percent = float('{:.2f}'.format((data.malware_label == 1).sum() / (data.malware_label == 0).sum()))
            model = OneClassSVM(gamma='scale', nu=nu_percent)
            trainX = X_train[y_train==0]
            model.fit(trainX)
            predict = model.predict(X_test)

            y_test[y_test == 1] = -1
            y_test[y_test == 0] = 1
            score = f1_score(y_test, predict, pos_label=-1)
        
        score_list.append(float(score))

    # Generate confusion matrix for output data
    if graph == 'confusion':
        conf_matrix = confusion_matrix(y_test, predict)
        
        if d_format == 'percent':
            matrix = conf_matrix/np.sum(conf_matrix)
            d_fmt = '.2%'
        else:
            matrix = conf_matrix
            d_fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix,
                    xticklabels=['Benign', 'Malware'],
                    yticklabels=['Benign', 'Malware'],
                    annot=True, fmt=d_fmt)
        plt.title('Confusion Matrix')
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
    
    return score_list

#def load_input (csv_data_file):
csv_data_file = 'test_train_data-all.csv'
dataset = pd.read_csv(csv_data_file)
# Remove columns filled with all 0 value (these will be statistically insignifant and will cause
# issues when using correlation methods of analysis)
data_no_z_cols = dataset.loc[:, (dataset != 0).any(axis=0)]