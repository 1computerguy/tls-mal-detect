#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score, classification_report, mean_absolute_error
from reduce_features import pca_output, random_forest, autoencoded_features
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression


def graph(x_data, y_data, graph):
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
    elif graph == 'margin':
        if load_model:
            label_val = np.array(label_test)
            feature_data = np.array(feature_data)
        else:
            label_val = np.array(feature_data.malware_label)
            feature_data = np.array(feature_data.drop(label, axis=1))

        plt.scatter(feature_data[:, 0], feature_data[:, 1], c=label_val, s=30, cmap=plt.cm.winter, alpha=0.5)
        ax = plt.gca()
        xlim = [-3, 3]
        ylim = [-3, 3]
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        XX, YY = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = svclassifier.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
        plt.show()
    elif graph == 'boundry':
        if load_model:
            label_val = label_test
        else:
            label_val = feature_data.malware_label
            feature_data = feature_data.drop(label, axis=1)

        plot_decision_regions(X=feature_data.values, y=label_val.values, clf=svclassifier, legend=2)
        plt.xlabel(feature_data.columns[0], size=14)
        plt.ylabel(feature_data.columns[1], size=14)
        plt.show()
    elif graph == 'auc':
        fpr, tpr, _ = roc_curve(label_test, predict)
        auc = roc_auc_score(label_test, predict)
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label='ROC Curve (area = {})'.format(str(auc)), color='darkorange')
        plt.title('Receiver Operating Characteristic (ROC curve)')
        plt.legend(loc='lower right')
        plt.show()
        
def mal_percent (data, percent_data):
    # Get a percentage of the malware data set to imbalance measurements for analysis
    malware = data[data.malware_label == 1]
    benign = data[data.malware_label == 0].reset_index(drop=True)
    percent = int(len(malware) * (percent_data / 100))
    malware = malware.sample(n=percent).reset_index(drop=True)
    
    return benign.append(malware).reset_index(drop=True)

def f_beta(beta, precision, recall):
    return (beta*beta + 1) * precision * recall / (beta * beta * precision + recall)

def svm_analysis(data, mal_data_percentage=100.0, load_model=False, model_type='svm', feature_reduction='pca', graph=None, d_format='digit'):
    '''
    Use a Support Vector Machine to classify malware and benign TLS traffic based on metadata
    gathered during the client/server handshake.
    '''
    label = 'malware_label'
    data = mal_percent(data, mal_data_percentage)

    #load_model = False
    if load_model:
        model_output_path = r'trained-model'
        model_file_name = r'{}\svm_model.pkl'.format(model_output_path)
        if os.path.exists(model_file_name):
            # Select feature reduction portion
            if feature_reduction == 'pca':
                feature_data = pca_output(data, 'malware_label', 2)
            elif feature_reduction == 'forest':
                feature_data = random_forest(data, 'malware_label', 1000)
            elif feature_reduction == 'none':
                mm_data = MinMaxScaler().fit_transform(data)
                feature_data = pd.DataFrame(mm_data, columns = data.columns)

            #label_test = pd.DataFrame(feature_data.malware_label.values)
            label_test = feature_data.malware_label.values
            feature_data = feature_data.drop(label, axis=1)
            svclassifier = joblib.load(model_file_name)
            #predict = svclassifier.predict(feature_data)
            for val in ['accuracy', 'precision', 'recall']:
                score = cross_val_score(svclassifier, feature_data, label_test, cv=10, scoring=val).mean()
                print("{}: {}".format(val, score))
                if val == 'precision':
                    prec = score
                elif val == 'recall':
                    rec = score
            print("F2 Score: {}".format(f_beta(2.0, prec, rec)))
            #score = fbeta_score(label_test, predict, beta=2.0)
        else:
            score = 'The loaded file path {} does not exist...'.format(model_file_name)
    else:
        # Select feature reduction portion
        if feature_reduction == 'pca':
            feature_data = pca_output(data, 'malware_label', 2)
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
            svclassifier = SVC(kernel='rbf', C=100, gamma=0.1, probability=True, random_state=42)
            svclassifier.fit(feature_train, label_train)
            predict = svclassifier.predict(feature_test)
            #for val in ['accuracy', 'precision', 'recall']:
            #    score = cross_val_score(svclassifier, feature_train, label_train, cv=10, scoring=val).mean()
            #    print("{}: {}".format(val, score))
            #    if val == 'precision':
            #        prec = score
            #    elif val == 'recall':
            #        rec = score
            #print("F2 Score: {}".format(f_beta(2.0, prec, rec)))
        
        elif model_type == 'one':
            # Set SVM hyperparameters in list for cross validation analysis
            #nu_percent = float('{:.2f}'.format((data.malware_label == 1).sum() / (data.index.stop)))
            nu_percent = 0.01
            #svclassifier = OneClassSVM(nu=nu_percent, kernel='rbf', gamma=0.1)
            svclassifier = OneClassSVM(nu=nu_percent, gamma=0.00001)
            predict = svclassifier.fit_predict(feature_train)
            mask = predict != -1
            #print(np.array(feature_train)[mask, :])
            #print(np.array(label_train)[mask])
            feature_train, label_train = np.array(feature_train)[mask, :], np.array(label_train)[mask]
            #predict = svclassifier.predict(feature_test)

            model = LinearRegression()
            model.fit(feature_train, label_train)
            predict = model.predict(feature_test)
            mae = mean_absolute_error(label_test, predict)
            print('MAE: {}'.format(mae))
            #label_test[label_test == 1] = 1
            #label_test[label_test == 0] = -1
            #score = f1_score(label_test, predict, pos_label=-1)

        save_model = True
        # Save model to model_file_name
        if save_model:
            # Output file name for trained model
            model_output_path = r'C:\Users\bryan\Desktop'
            model_file_name = r'{}\svm_model.pkl'.format(model_output_path)
            if not os.path.exists(model_output_path):
                os.mkdir(model_output_path)

            joblib.dump(svclassifier, model_file_name)


    
    #print('Accuracy: {}'.format(accuracy_score(label_test, predict)))
    #print('Precision: {}'.format(precision_score(label_test.values, predict, average='micro')))
    #print('Recall: {}'.format(recall_score(label_test.values, predict, average='micro')))
    #print('F0.5 Score: {}'.format(fbeta_score(label_test, predict.round(), beta=0.5, average='micro')))
    #print('F1 Score: {}'.format(f1_score(label_test, predict.round(), average='micro')))
    #print('F2 Score: {}'.format(fbeta_score(label_test, predict, beta=2.0, average='micro')))

#def load_input (csv_data_file):
csv_data_file = r'test-train-data\test\test_train_data-all.csv'
new_csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'

dataset = pd.read_csv(csv_data_file)
new_dataset = pd.read_csv(new_csv_data_file)
full_dataset = pd.concat([dataset, new_dataset], axis=0).reset_index()
full_dataset = full_dataset.drop(['index'], axis=1)
# Remove columns filled with al2l 0 value (these will be statistically insignifant and may cause
# issues when using correlation methods of analysis)
#data_no_z_cols = dataset.loc[:, (dataset != 0).any(axis=0)]
#new_data_no_z_cols = new_dataset.loc[:, (new_dataset != 0).any(axis=0)]
