#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score
from reduce_features import pca_output, random_forest, autoencoded_features

def mal_percent (data, percent_data):
    # Get a percentage of the malware data set to imbalance measurements for analysis
    malware = data[data.malware_label == 1]
    benign = data[data.malware_label == 0].reset_index(drop=True)
    percent = int(len(malware) * (percent_data / 100))
    malware = malware.sample(n=percent).reset_index(drop=True)
    
    return benign.append(malware).reset_index(drop=True)

def svm_analysis(data, label, mal_data_percentage=100.0, load_model=False, model_type='svm', feature_reduction='pca', graph=None, d_format='digit'):
    '''
    Use a Support Vector Machine to classify malware and benign TLS traffic based on metadata
    gathered during the client/server handshake.
    '''
    data = mal_percent(data, mal_data_percentage)

    #load_model = False
    if load_model:
        model_output_path = r'trained-model'
        model_file_name = r'{}\svm_model.pkl'.format(model_output_path)
        if os.path.exists(model_file_name):
            # Select feature reduction portion
            if feature_reduction == 'pca':
                feature_data = pca_output(data, 'malware_label')
            elif feature_reduction == 'forest':
                feature_data = random_forest(data, 'malware_label', 1000)
            elif feature_reduction == 'none':
                mm_data = MinMaxScaler().fit_transform(data)
                feature_data = pd.DataFrame(mm_data, columns = data.columns)

            label_test = feature_data.malware_label
            feature_data = feature_data.drop(label, axis=1)
            loaded_model = joblib.load(model_file_name)
            #print(loaded_model)
            predict = loaded_model.predict(feature_data)
            score = fbeta_score(label_test, predict, beta=2.0)
        else:
            score = 'The loaded file path {} does not exist...'.format(model_file_name)
    else:
        # Select feature reduction portion
        if feature_reduction == 'pca':
            feature_data = pca_output(data, 'malware_label')
        elif feature_reduction == 'forest':
            feature_data = random_forest(data, 'malware_label', 1000)
        elif feature_reduction == 'auto':
            feature_data = autoencoded_features(data, 'malware_label', 10)
        elif feature_reduction == 'none':
            mm_data = MinMaxScaler().fit_transform(data)
            feature_data = pd.DataFrame(mm_data, columns = data.columns)

        tt_features = feature_data.drop(label, axis=1)
        tt_labels = feature_data[label]

        feature_train, feature_test, label_train, label_test = train_test_split(tt_features, tt_labels, test_size=0.20)

        # Select type of SVM to use, standard Support vector or One-class SVM
        if model_type == 'svm':
            # Set SVM hyperparameters in list for cross validation analysis
            param = [
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1e-1, 1e-2, 1e-3, 1e-4]
                },
                {
                    'kernel': ['linear'],
                    'C': [0.1, 1, 10, 100]
                }
            ]
            svclassifier = SVC(kernel='rbf', C=100, gamma=0.1, probability=True)
            #svclassifier = SVC(probability=True)
            #cross_validate = GridSearchCV(svclassifier, param, cv=10, n_jobs=5, verbose=3)
            #cross_validate.fit(feature_train, label_train)
            svclassifier.fit(feature_train, label_train)
            predict = svclassifier.predict(feature_test)
            #predict = cross_validate.predict(feature_test)
            score = fbeta_score(label_test, predict, beta=2.0)

        elif model_type == 'one':
            # Set SVM hyperparameters in list for cross validation analysis
            nu_percent = float('{:.2f}'.format((data.malware_label == 1).sum() / (data.malware_label == 0).sum()))
            param = [
                {
                    'kernel': ['rbf'],
                    'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                    'nu': [nu_percent]
                }
            ]
            svclassifier = OneClassSVM()
            cross_validate = GridSearchCV(svclassifier, param, cv=10, n_jobs=5)
            train_features = feature_train[feature_train==0]
            cross_validate.fit(train_features)
            predict = cross_validate.predict(feature_test)

            label_test[label_test == 1] = -1
            label_test[label_test == 0] = 1
            score = f1_score(label_test, predict, pos_label=-1)

        save_model = True
        # Save model to model_file_name
        if save_model:
            # Output file name for trained model
            model_output_path = r'C:\Users\bryan\Desktop'
            model_file_name = r'{}\svm_model.pkl'.format(model_output_path)
            if not os.path.exists(model_output_path):
                os.mkdir(model_output_path)

            joblib.dump(svclassifier, model_file_name)
            #with open(model_file_name, 'wb') as f:
            #    dump(pure_predict, f)

    # Generate confusion matrix for output data
    if graph == 'confusion':
        conf_matrix = confusion_matrix(label_test, predict)
        
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
    
    return score

#def load_input (csv_data_file):
csv_data_file = r'test-train-data\test\test_train_data-all.csv'
new_csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'

dataset = pd.read_csv(csv_data_file)
new_dataset = pd.read_csv(new_csv_data_file)
# Remove columns filled with all 0 value (these will be statistically insignifant and may cause
# issues when using correlation methods of analysis)
#data_no_z_cols = dataset.loc[:, (dataset != 0).any(axis=0)]
#new_data_no_z_cols = new_dataset.loc[:, (new_dataset != 0).any(axis=0)]
