#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os
import umap

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import mean_absolute_error
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from numpy.random import seed

def create_graph(x_data, y_data, graph, model=None, ae=False, label_data=None):
    '''
    Graph an ML model's training or analysis output to visualize its efficacy and functionality
    '''
    # Generate confusion matrix for output data
    if graph == 'confusion':
        if ae:
            pred_x = [1 if e > y_data else 0 for e in x_data['Loss_mae'].values]
            conf_matrix = confusion_matrix(label_data, pred_x)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix,
                        xticklabels=['Benign', 'Malware'],
                        yticklabels=['Benign', 'Malware'],
                        annot=True, fmt='d')
            plt.title('Confusion Matrix')
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
        else:
            conf_matrix = confusion_matrix(y_data, x_data)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix,
                        xticklabels=['Benign', 'Malware'],
                        yticklabels=['Benign', 'Malware'],
                        annot=True, fmt='d')
            plt.title('Confusion Matrix')
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
    # Create an SVM margin graph to visualize the Maximal Margin
    elif graph == 'margin':
        x_data = np.array(x_data)

        plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data.values, s=30, cmap=plt.cm.winter, alpha=0.5)
        ax = plt.gca()
        xlim = [-3, 3]
        ylim = [-3, 3]
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        XX, YY = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = model.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='k')
    # Create a colored Margin graph
    elif graph == 'boundary':
        x_data = np.array(x_data) if isinstance(x_data, pd.DataFrame) else x_data
        plot_decision_regions(X=x_data, y=y_data.values, clf=model, legend=2)
        plt.xlabel("Component 1", size=14)
        plt.ylabel("Component 2", size=14)
    # Graph the AUC
    elif graph == 'auc':
        fpr, tpr, _ = roc_curve(y_data, x_data)
        auc = roc_auc_score(y_data, x_data)
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label='ROC Curve (area = {})'.format(str(auc)), color='darkorange')
        plt.title('Receiver Operating Characteristic (ROC curve)')
        plt.legend(loc='lower right')
    ######################
    # Below graphs are specifically for the Autoencoder
    # Graph the AE loss curve
    elif graph == 'loss':
        plt.plot(x_data,
                'b',
                label='Training loss')
        plt.plot(y_data,
                'r',
                label='Validation loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss, [mse]')
        plt.ylim([0,.014])
    # Show the Mean Absolute Error graph (used for the threshold)
    # to determine the number of standard deviations above the peak of the 
    # curve that the learning reaches "0"
    elif graph == 'mae':
        X_pred = model.predict(np.array(x_data))
        X_pred = pd.DataFrame(X_pred, columns=x_data.columns)
        X_pred.index = x_data.index

        pred = pd.DataFrame(index=x_data.index)
        pred['Loss_mae'] = np.mean(np.abs(X_pred-x_data), axis = 1)
        plt.figure()
        sns.distplot(pred['Loss_mae'],
                    bins = 10, 
                    kde= True,
                    color = 'blue')
        plt.xlim([0.0,.02])
    # Simple time-based plot of threshold and data points
    elif graph == 'thresh':
        x_data.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e0], color = ['blue','red'])
    # Scatter chart offering clearer, easier to understand malicious vs. benign data
    # representation with threshold marker
    elif graph == 'scatter':
        sns.scatterplot( x=x_data.index, y='Loss_mae', data=x_data, palette='plasma', hue='Anomaly')
        plt.axhline(y=y_data)

    plt.show()

def f_beta(beta, precision, recall):
    '''
    Calculate the F score indicating the beta value to use 0.5, 1, or 2
    '''
    return (beta*beta + 1) * precision * recall / (beta * beta * precision + recall)

def save_model(model, filename, ae=False):
    '''
    Save a model to disk for later analysis
    '''
    model_output_path = os.path.split(os.path.abspath(filename))[0]
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)

    if ae:
        model.save(filename)
    else:
        joblib.dump(model, filename)

def import_model(filename, ae=False):
    '''
    Import a pre-trained model for analysis
    '''
    if os.path.exists(filename):
        if ae:
            loaded_model = load_model(filename)
        else:
            loaded_model = joblib.load(filename)
    else:
        loaded_model = 'The file path {} does not exist...'.format(filename)
    
    return loaded_model

def AE_threshold(train_dist, pred_dist, extreme=False):
    '''
    Calculate the Autoencoder threshold
        - Above the threshold marks anomalous/malicious data
        - Below the threshold marks normal/benign data
    '''
    k = 4. if extreme else 3.
    train_thresh = np.mean(np.mean(np.abs(train_dist), axis = 1))
    pred_thresh = np.mean(np.mean(np.abs(pred_dist), axis = 1))
    threshold = np.abs(pred_thresh - train_thresh) * k
    return threshold

def svm(data, scores=False, load_model=False, graph=None, feature_reduce=False):
    '''
    Use a Support Vector Machine to classify malware and benign TLS traffic based on metadata
    gathered during the client/server handshake.
    '''
    label = 'malware_label'
    tt_features = data.drop(label, axis=1)
    tt_labels = data[label]

    # Feature reduction to 2 components required for margin and boundary graphs
    if feature_reduce == 'umap':
        tt_features = umap.UMAP(n_neighbors=5, min_dist=0.01, n_components=2).fit_transform(tt_features)
    elif feature_reduce == 'pca':
        pca = PCA(n_components=2).fit_transform(tt_features)
        tt_features = pd.DataFrame(pca)

    feature_train, feature_test, label_train, label_test = train_test_split(tt_features, tt_labels, test_size=0.20)

    # Select type of SVM to use, standard Support vector or One-class SVM
    # Set SVM hyperparameters in list for cross validation analysis
    svclassifier = SVC(kernel='rbf', C=100, gamma=0.1, probability=True, random_state=42)
    svclassifier.fit(feature_train, label_train)
    if scores:
        for val in ['accuracy', 'precision', 'recall']:
            score = cross_val_score(svclassifier, feature_train, label_train, cv=10, scoring=val).mean()
            print("{}: {}".format(val, score))
            if val == 'precision':
                prec = score
            elif val == 'recall':
                rec = score
        print("F2 Score: {}".format(f_beta(2.0, prec, rec)))

    svm_pred = svclassifier.predict(feature_test)

    if graph == 'margin' or graph == 'boundary':
        create_graph(tt_features, tt_labels, graph, svclassifier)
    elif graph:
        create_graph(svm_pred, label_test, graph, svclassifier)

def oc_svm(data, mal_percent, test_percent, scores=False, graph=None):
    '''
    Use a One-Class Support Vector Machine to classify malware and benign TLS traffic based
    on metadata gathered during the client/server handshake.
    '''
    # Set nu hyperparameter and test percentage for test_train_split
    nu_percent = mal_percent / 100
    #gamma_val = 0.0001
    # Baseline gamma for rbf kernel
    gamma_val = 0.000000001
    test_percent = test_percent / 100
    label = 'malware_label'

    oc_benign = data[data.malware_label == 1]

    oc_b_train, oc_b_test = train_test_split(oc_benign, test_size=test_percent, random_state=1)
    oc_b_train = oc_b_train.drop([label], axis=1)

    oc_malware = data[data.malware_label == -1]

    oc_test = oc_b_test.append(oc_malware)
    oc_test_label = oc_test.malware_label
    oc_test = oc_test.drop([label], axis=1)

    svclassifier = OneClassSVM(nu=nu_percent, kernel='rbf', gamma=gamma_val)
    svclassifier.fit(oc_b_train)
    if scores:
        for val in ['accuracy', 'precision', 'recall']:
            score = cross_val_score(svclassifier, oc_test, oc_test_label, cv=10, scoring=val).mean()
            print("{}: {}".format(val, score))
            if val == 'precision':
                prec = score
            elif val == 'recall':
                rec = score
        print("F2 Score: {}".format(f_beta(2.0, prec, rec)))
    
    oc_pred = svclassifier.predict(oc_test)

    if graph:
        create_graph(oc_pred, oc_test_label, graph, svclassifier)

def ae(data, scores=False, save=False, load=False, graph=None):
    '''
    Use an Autoencoder Neural Network to classify malware and benign TLS traffic based
    on metadata gathered during the client/server handshake.
    '''
    NUM_EPOCHS=20
    BATCH_SIZE=64
    label = 'malware_label'
    extreme = False
    seed(1)
    set_seed(2)
    act_func = 'elu'
    model_file = r'trained-model\ae_classifier.h5'
    save_file = r'C:\Users\bryan\Desktop\ae_classifier.h5'

    label_data = data.malware_label

    if load:
        data = data.drop(label, axis=1)

        model = import_model(model_file, True)
        predictions = model.predict(np.array(data))
        predictions = pd.DataFrame(predictions, columns=data.columns)
        predictions.index = data.index
        
        threshold = AE_threshold(data, predictions, extreme)
        scored = pd.DataFrame(index=data.index)
        scored['Loss_mae'] = np.mean(np.abs(predictions-data), axis=1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    else:
        x_train = data[data.malware_label == 0]
        x_train = x_train.drop(label, axis=1)
        x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train),
                                columns=x_train.columns,
                                index=x_train.index)
        x_test = data[data.malware_label == 1]
        x_test = x_test.drop(label, axis=1)
        x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test),
                                columns=x_test.columns,
                                index=x_test.index)
        begin_end_length = x_train.shape[1]
        stage_two_length = 100
        stage_three_length = 2

        model=Sequential()
        # First hidden layer, connected to input vector X. 
        model.add(Dense(stage_two_length, activation=act_func,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.0),
                        input_shape=(begin_end_length,)
                    )
                )

        model.add(Dense(stage_three_length , activation=act_func,
                        kernel_initializer='glorot_uniform'))

        model.add(Dense(stage_two_length , activation=act_func,
                        kernel_initializer='glorot_uniform'))

        model.add(Dense(begin_end_length,
                        kernel_initializer='glorot_uniform'))

        model.compile(loss='mse',optimizer='adam')

        history = model.fit(np.array(x_train), np.array(x_train),
                        batch_size=BATCH_SIZE, 
                        epochs=NUM_EPOCHS,
                        validation_split=0.05,
                        verbose = 0)

        if save:
            save_model(model, save_file, True)

        predictions = model.predict(np.array(x_test))
        predictions = pd.DataFrame(predictions, 
                            columns=x_test.columns)
        predictions.index = x_test.index

        scored = pd.DataFrame(index=x_test.index)
        threshold = AE_threshold(x_train, predictions, extreme)
        scored['Loss_mae'] = np.mean(np.abs(predictions-x_test), axis = 1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

        predictions_train = model.predict(np.array(x_train))
        predictions_train = pd.DataFrame(predictions_train, 
                            columns=x_train.columns)
        predictions_train.index = x_train.index

        scored_train = pd.DataFrame(index=x_train.index)
        scored_train['Loss_mae'] = np.mean(np.abs(predictions_train-x_train), axis = 1)
        scored_train['Threshold'] = threshold
        scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
        scored = pd.concat([scored_train, scored]).sort_index()

    if scores:
        print('Accuracy: {}'.format(accuracy_score(label_data, scored['Anomaly'])))
        print('Precision: {}'.format(precision_score(label_data, scored['Anomaly'])))
        print('Recall: {}'.format(recall_score(label_data, scored['Anomaly'])))
        print('F2 Score: {}'.format(fbeta_score(label_data, scored['Anomaly'], beta=2.0)))
    
    if graph == 'loss':
        create_graph(history.history['loss'], history.history['val_loss'], graph)
    elif graph == 'scatter':
        create_graph(scored, threshold, graph, model)
    elif graph == 'confusion':
        create_graph(scored, threshold, graph, ae=True, label_data=label_data)
    elif graph:
        create_graph(scored, label_data, graph, model)

def get_data(sample_size, mal_percent=20, test_percent=20, occ=False):
    csv_data_file = r'test-train-data\test\test_train_data-all.csv'
    new_csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'

    dataset = pd.read_csv(csv_data_file)
    new_dataset = pd.read_csv(new_csv_data_file)
    full_dataset = pd.concat([dataset, new_dataset], axis=0).reset_index(drop=True)

    mm_data = MinMaxScaler().fit_transform(full_dataset)
    full_dataset = pd.DataFrame(mm_data, columns=full_dataset.columns)
    full_dataset.malware_label = full_dataset.malware_label.astype(int)

    if occ:
        label = 'malware_label'
        ben = 1
        mal = -1
        full_dataset.loc[full_dataset[label] == 1, label] = mal
        full_dataset.loc[full_dataset[label] == 0, label] = ben

        benign = full_dataset[full_dataset.malware_label == ben]
        malware = full_dataset[full_dataset.malware_label == mal]
        
        test_size = int((test_percent / 100) * sample_size)
        mal_size = int((mal_percent / 100) * test_size)

        malware = malware.sample(n=mal_size, random_state=1)
        benign = benign.sample(n=(sample_size - mal_size), random_state=1)
        
        sampled_data = benign.append(malware).reset_index(drop=True)
    else:
        ben = 0
        mal = 1

        benign = full_dataset[full_dataset.malware_label == ben]
        malware = full_dataset[full_dataset.malware_label == mal]

        mal_size = int((mal_percent / 100) * sample_size)

        # Prevent malware sample size from being larger than actual sample size
        if mal_size > malware.shape[0]:
            mal_size = malware.shape[0]

        malware = malware.sample(n=mal_size, random_state=1)
        benign = benign.sample(n=(sample_size - mal_size), random_state=1)
        
        sampled_data = benign.append(malware).reset_index(drop=True)

    sampled_data = sampled_data.sample(frac=1).reset_index(drop=True)

    return sampled_data
