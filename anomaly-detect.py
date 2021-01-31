#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from numpy.random import seed

def create_graph(x_data, y_data, graph, model=None, ae=False, occ=False, label_data=None, export=False, graph_file=Path('./graph/graph.png')):
    '''
    Graph an ML model's training or analysis output to visualize its efficacy and functionality
        - ae and label_data arguments are used to signify Autoencoder graphs
    '''
    # Generate confusion matrix for output data
    if graph == 'confusion':
        if ae:
            pred_x = [1 if e > y_data else 0 for e in x_data['Loss_mae'].values]
            conf_matrix = confusion_matrix(label_data, pred_x)
        else:
            if occ:
                ben = 0
                mal = 1
                x_data[x_data == 1] = ben
                x_data[x_data == -1] = mal
                y_data = y_data.values
                y_data[y_data == 1] = ben
                y_data[y_data == -1] = mal

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
        xlim = [-15, 15]
        ylim = [-15, 15]
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
        y_data = np.array(y_data).astype(np.integer)
        plot_decision_regions(X=x_data, y=y_data, clf=model, legend=2)
        plt.xlabel("Component 1", size=12)
        plt.ylabel("Component 2", size=12)
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
    ###################################################
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
        pred['Loss_mae'] = np.mean(np.abs(X_pred - x_data), axis = 1)
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

    if export:
        plt.save(graph_file)
    else:
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

def svm(data, scores=False, save=False, load=False, filename=Path('./models/svm.pkl'), graph=None, graph_file=None):
    '''
    Use a Support Vector Machine to classify malware and benign TLS traffic based on metadata
    gathered during the client/server handshake.
    '''
    label = 'malware_label'
    tt_features = data.drop(label, axis=1)
    tt_labels = data[label]

    # Feature reduction to 2 components required for margin and boundary graphs
    if graph == 'margin' or graph == 'boundary':
        pca = PCA(n_components=2).fit_transform(tt_features)
        tt_features = pd.DataFrame(pca)

    # Laod model from file
    if load:
        svclassifier = import_model(filename.absolute())
        feature_test = tt_features
        label_test = tt_labels

    # Train model
    else:
        feature_train, feature_test, label_train, label_test = train_test_split(tt_features, tt_labels, test_size=0.20)
        svclassifier = SVC(kernel='rbf', C=100, gamma=0.1, probability=True, random_state=42)
        svclassifier.fit(feature_train, label_train)

        # Save model to file
        if save:
            save_model(svclassifier, filename)

    # Perform n-fold cross validation and calculate the mean score across the cv=## folds
    if scores:
        print('\nCalculating SVM scores...')
        for val in ['accuracy', 'precision', 'recall']:
            score = cross_val_score(svclassifier, feature_test, label_test, cv=10, scoring=val).mean()
            print("{}: {}".format(val, score))
            if val == 'precision':
                prec = score
            elif val == 'recall':
                rec = score
        print("F2 Score: {}".format(f_beta(2.0, prec, rec)))

    svm_pred = svclassifier.predict(feature_test)

    export = True if graph_file else False

    if graph == 'margin' or graph == 'boundary':
        create_graph(tt_features, tt_labels, graph, svclassifier, export, graph_file)
    elif graph:
        create_graph(svm_pred, label_test, graph, export, graph_file)

    #return svm_pred, tt_labels

def oc_svm(data, mal_percent, scores=False, save=False, load=False, filename=Path('./models/oc-svm.pkl'), graph=None, graph_file=None):
    '''
    Use a One-Class Support Vector Machine to classify malware and benign TLS traffic based
    on metadata gathered during the client/server handshake.
    '''
    # Set nu and gamma hyperparameters and test percentage
    test_percent = 0.20
    nu_value = (mal_percent / 100) * test_percent
    gamma_val = 0.1
    label = 'malware_label'

    #if graph == 'margin' or graph == 'boundary':
        # Feature reduction to 2 components required for margin and boundary graphs
        # OC-SVM does not graph well using any of the attempted feature reduction techniques.
        # I attempted PCA, Autoencoder, T-SNE, UMAP, Factor Analysis, Random Forest. The PCA
        # and AE are left below merely for reference.

        #label_data = data.malware_label
        #feature_data = data.drop(label, axis=1)

        #pca = PCA(n_components=2).fit_transform(feature_data)
        #data = pd.DataFrame(pca)
        #data = pd.concat([data, label_data], axis=1)

        #ae = Sequential()
        #ae.add(Dense(100, activation='elu',
        #            kernel_initializer='glorot_uniform',
        #            input_shape=(feature_data.shape[1],),
        #            kernel_regularizer=regularizers.l2(0.0)))
        #ae.add(Dense(2, activation='elu', name='bottleneck', kernel_initializer='glorot_uniform'))
        #ae.add(Dense(100, activation='elu', kernel_initializer='glorot_uniform'))
        #ae.add(Dense(feature_data.shape[1], activation='sigmoid', kernel_initializer='glorot_uniform'))
        #ae.compile(loss='mse',optimizer='adam')
        #ae.fit(np.array(feature_data), np.array(feature_data), batch_size=64, epochs=20, verbose=0)
        #encoder = Model(ae.input, ae.get_layer('bottleneck').output)
        #ae_data = encoder.predict(feature_data)
        #data = pd.DataFrame(ae_data)
        #data = pd.concat([data, label_data], axis=1)

    # Laod model from file
    if load:
        svclassifier = import_model(filename)
        oc_test = data.drop(label, axis=1)
        oc_test_label = data[label]

    # Train model
    else:
        oc_benign = data[data.malware_label == 1]
        oc_malware = data[data.malware_label == -1]

        oc_b_train, oc_b_test = train_test_split(oc_benign, test_size=test_percent, random_state=1)
        oc_b_train = oc_b_train.drop(label, axis=1)

        oc_test = oc_b_test.append(oc_malware)
        oc_test_label = oc_test.malware_label
        oc_test = oc_test.drop(label, axis=1)

        svclassifier = OneClassSVM(nu=nu_value, kernel='rbf', gamma=gamma_val)
        svclassifier.fit(oc_b_train)

        # Save model to file
        if save:
            save_model(svclassifier, filename)

    # Perform n-fold cross validation and calculate the mean score across the cv=## folds
    if scores:
        print('\nCalculating OC-SVM scores...')
        for val in ['accuracy', 'precision', 'recall']:
            score = cross_val_score(svclassifier, oc_test, oc_test_label, cv=10, scoring=val).mean()
            print("{}: {}".format(val, score))
            if val == 'precision':
                prec = score
            elif val == 'recall':
                rec = score
        print("F2 Score: {}".format(f_beta(2.0, prec, rec)))
        
    oc_pred = svclassifier.predict(oc_test)

    export = True if graph_file else False

    # You can uncomment the below if statement (and change the second if graph to an elif graph)
    # ONLY if you enable one of the feature reduction techniques above - either PCA or the AE
    #if graph == 'margin' or graph == 'boundary':
    #    create_graph(oc_test, oc_test_label, graph, svclassifier, export, graph_file)
    if graph == 'confusion' or graph == 'auc':
        create_graph(oc_pred, oc_test_label, graph, svclassifier, export, graph_file, occ=True)
    else:
        print('You need to uncomment one of the feature reduction techniques in this function to use that graph type...')

def ae(data, scores=False, save=False, load=False, filename=Path('./models/ae.h5'), graph=None, graph_file=None):
    '''
    Use an Autoencoder Neural Network to classify malware and benign TLS traffic based
    on metadata gathered during the client/server handshake.
    '''
    seed(1)
    set_seed(2)
    NUM_EPOCHS=200
    BATCH_SIZE=32
    label = 'malware_label'
    act_func = 'elu'
    label_data = data.malware_label

    # Laod model from file
    if load:
        data = data.drop(label, axis=1)

        model = import_model(filename, True)
        predictions = model.predict(np.array(data))
        predictions = pd.DataFrame(predictions, columns=data.columns)
        predictions.index = data.index
        
        # Calculate threshold and provide anomaly output predictions as a dataframe
        threshold = AE_threshold(data, predictions)
        scored = pd.DataFrame(index=data.index)
        scored['Loss_mae'] = np.mean(np.abs(predictions-data), axis=1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

    # Train model
    else:
        x_train = data[data.malware_label == 0]
        x_train = x_train.drop(label, axis=1)
        x_test = data[data.malware_label == 1]
        x_test = x_test.drop(label, axis=1)
        
        begin_end_length = x_train.shape[1]
        stage_two_length = 300
        stage_three_length = 100
        stage_four_length = 2

        # Build AE network
        model = Sequential()

        model.add(Dense(stage_two_length, activation=act_func,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(0.0),
                        input_shape=(begin_end_length,)
                    )
                )

        model.add(Dense(stage_three_length , activation=act_func,
                        kernel_initializer='glorot_uniform'))
        model.add(Dense(stage_four_length , activation=act_func,
                        kernel_initializer='glorot_uniform'))
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

        # Save model to file
        if save:
            save_model(model, filename, True)

        predictions = model.predict(np.array(x_test))
        predictions = pd.DataFrame(predictions, 
                            columns=x_test.columns)
        predictions.index = x_test.index

        # Calculate scores if conducting training and validation
        #   - First set of measurements are for benign traffic
        #   - Second set of measurements are for predictions based on first set
        #     and for detecting malware
        scored = pd.DataFrame(index=x_test.index)
        threshold = AE_threshold(x_train, predictions, True)
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

    # Cross_val_score does not support AE model, so we calculate scores individually
    # Did not find a suitable method of N-fold cross validation...
    if scores:
        print('\nPrinting Autoencoder scores...')
        print('Accuracy: {}'.format(accuracy_score(label_data, scored['Anomaly'])))
        print('Precision: {}'.format(precision_score(label_data, scored['Anomaly'])))
        print('Recall: {}'.format(recall_score(label_data, scored['Anomaly'])))
        print('F2 Score: {}'.format(fbeta_score(label_data, scored['Anomaly'], beta=2.0)))

    export = True if graph_file else False
    if graph == 'loss':
        create_graph(history.history['loss'], history.history['val_loss'], graph, export, graph_file)
    elif graph == 'scatter':
        create_graph(scored, threshold, graph, model)
    elif graph == 'confusion':
        create_graph(scored, threshold, graph, export, graph_file, ae=True, label_data=label_data)
    elif graph == 'mae':
        create_graph(x_train, label_data, graph, model, export, graph_file)
    elif graph:
        create_graph(scored, label_data, graph, model, export, graph_file)

def get_data(sample_size, mal_percent=20, test_percent=20, occ=False):
    rand_state_val = 42
    csv_data_file = Path('./test-train-data/test_train_data.csv')
    full_dataset = pd.read_csv(csv_data_file)

    # Scale data to 0-1 value for more efficient ML analysis
    mm_data = MinMaxScaler().fit_transform(full_dataset)
    full_dataset = pd.DataFrame(mm_data, columns=full_dataset.columns)

    # If model is OC-SVM convert label values to 1 and -1 (this is how OC-SVM
    # outputs predictions, so validation requires these values
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

        # Prevent malware sample size from being larger than actual sample size
        if mal_size > malware.shape[0]:
            mal_size = malware.shape[0]

        malware = malware.sample(n=mal_size, random_state=rand_state_val)

        # Prevent total sample size from being larger than actual sample size
        total_sample_size = sample_size - mal_size
        if total_sample_size > benign.shape[0]:
            total_sample_size = benign.shape[0]

        benign = benign.sample(n=total_sample_size, random_state=rand_state_val)
        sampled_data = benign.append(malware).reset_index(drop=True)
    # If not OC-SVM then provide desitnated testing and malware distributions
    else:
        ben = 0
        mal = 1

        benign = full_dataset[full_dataset.malware_label == ben]
        malware = full_dataset[full_dataset.malware_label == mal]

        mal_size = int((mal_percent / 100) * sample_size)

        # Prevent malware sample size from being larger than actual sample size
        if mal_size > malware.shape[0]:
            mal_size = malware.shape[0]

        malware = malware.sample(n=mal_size, random_state=rand_state_val)

        # Prevent total sample size from being larger than actual sample size
        total_sample_size = sample_size - mal_size
        if total_sample_size > benign.shape[0]:
            total_sample_size = benign.shape[0]

        benign = benign.sample(n=total_sample_size, random_state=rand_state_val)
        sampled_data = benign.append(malware).reset_index(drop=True)

    sampled_data = sampled_data.sample(frac=1).reset_index(drop=True)

    return sampled_data

def main():
    '''
    Execute above functions and run through the various ML models outlined in the paper:
        Malware Detection in Encrypted TLS Traffic Through Machine Learning
    '''

    parser = ArgumentParser(description='Run an ML model to analyse TLS data for malicious activity.',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--data', action='store', dest='data_size', default=0,
                        help='Data sample size to analyze', type=int, required=True)
    parser.add_argument('-m', '--malware', action='store', dest='malware_size', default=20,
                        help='Percentage of dataset that is Malware', type=float, required=False)
    parser.add_argument('-t', '--test', action='store', dest='test_size', default=20,
                        help='Percentage of dataset to use for validation', type=float, required=False)
    parser.add_argument('-o', '--model', action='store', dest='ml_model', 
                        help='''Machine Learning model to use. Acceptable values are:
- ae = Autoencoder
- svm = Support Vector Machine
- oc-svm = One-Class SVM''',
                        required=False)
    parser.add_argument('-s', '--save', action='store_true', dest='save_model', default=False,
                        help='Save the trained model - REQUIRES the -f/--file option', required=False)
    parser.add_argument('-l', '--load', action='store_true', dest='load_model', default=False,
                        help='Evaluate data against a trained model - REQUIRES the -f/--file option', required=False)
    parser.add_argument('-f', '--file', action='store', dest='file', default=None,
                        help='Save/Load file path', required=False)
    parser.add_argument('-r', '--scores', action='store_true', dest='scores', default=False,
                        help='Print 10-fold cross-validated Accuracy, Recall, Precision, and F2 scores', required=False)
    parser.add_argument('-g', '--graph', action='store', dest='graph', default=None,
                        help='''Visualize the modeled dataset. Acceptable values are:
SVM and OC-SVM graphs:
    - confusion
    - margin (SVM only)
    - boundary (SVM only)
    - auc
Autoencoder graphs:
    - confusion
    - loss
    - mae
    - thresh
    - scatter''',
                        required=False)
    parser.add_argument('-p', '--print', action='store_true', dest='print_data', default=False,
                        help='Print dataset', required=False)
    parser.add_argument('-e', '--export', action='store_true', dest='export', default=False,
                        help='This will save the graph to a file - REQUIRED if running in a container', required=False)

    options = parser.parse_args()

    if (options.save_model or options.load_model) and not options.file:
        print('If you want to save or load a model, you must also use the -f or\n--file option and provide the location of the file.')
        quit()

    data_size = options.data_size
    malware_size = options.malware_size
    test_size = options.test_size
    save = options.save_model
    load = options.load_model   
    scores = options.scores
    graph = options.graph
    print_data = options.print_data
    model = options.ml_model
    export_graph = options.export

    occ = True if model == 'oc-svm' else False
    filename = Path(options.file) if options.file else None
    
    graph_file = Path('/detect/graph/{}-{}.png'.format(model, graph)) if export_graph else None

    if load and not filename.exists():
        print('\n The file {} cannot be found... Please check your spelling and try again'.format(filename))
        quit()

    dataset = get_data(data_size, malware_size, test_size, occ)

    if print_data:
        print(dataset)
    
    if model == 'ae':
        ae(dataset, scores, save, load, filename, graph, graph_file)
    elif model == 'svm':
        svm(dataset, scores, save, load, filename, graph, graph_file)
    elif model == 'oc-svm':
        oc_svm(dataset, malware_size, scores, save, load, filename, graph, graph_file)
    elif model:
        print('\nPlease choose a model of type ae, svm, or oc-svm... To get help using this script use the -h or --help option')
        quit()

if __name__ == '__main__':
    main()

