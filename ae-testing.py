import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score, classification_report

def graph(x_data, y_data, graph):
    if graph == 'loss':
        plt.plot(x_data,
                'b',
                label='Training loss')
        plt.plot(x_data,
                'r',
                label='Validation loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss, [mse]')
        plt.ylim([0,.008])
        plt.show()
    elif graph == 'mae':
        X_pred = model.predict(np.array(x_train))
        X_pred = pd.DataFrame(X_pred, columns=x_train.columns)
        X_pred.index = x_train.index

        pred = pd.DataFrame(index=x_train.index)
        pred['Loss_mae'] = np.mean(np.abs(X_pred-x_train), axis = 1)
        plt.figure()
        sns.distplot(pred['Loss_mae'],
                    bins = 10, 
                    kde= True,
                    color = 'blue')
        plt.xlim([0.0,.02])
        plt.show()
    elif graph == 'thresh':
        scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e0], color = ['blue','red'])
        plt.show()
    elif graph == 'confusion':
        pred_x = [1 if e > threshold else 0 for e in scored['Loss_mae'].values]
        conf_matrix = confusion_matrix(label_data, pred_x)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix,
                    xticklabels=['Benign', 'Malware'],
                    yticklabels=['Benign', 'Malware'],
                    annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
    elif graph == 'scatter':
        sns.scatterplot( x=scored.index, y='Loss_mae', data=scored, palette='plasma', hue='Anomaly')
        plt.axhline(y=threshold)
        plt.show()
    elif graph == 'roc':
        plt.plot(label_data, scored['Anomaly'])
        plt.show()
    elif graph == 'auc':
        fpr, tpr, _ = roc_curve(label_data, scored['Anomaly'])
        auc = roc_auc_score(label_data, scored['Anomaly'])
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label='ROC Curve (area = {})'.format(str(auc)), color='darkorange')
        plt.title('Receiver Operating Characteristic (ROC curve)')
        plt.legend(loc='lower right')
        plt.show()


def AE_threshold(train_dist, pred_dist, extreme=False):
    k = 4. if extreme else 3.
    train_thresh = np.mean(np.mean(np.abs(train_dist), axis = 1))
    pred_thresh = np.mean(np.mean(np.abs(pred_dist), axis = 1))
    threshold = np.abs(pred_thresh - train_thresh) * k
    return threshold

def create_model(data_shape):
    seed(1)
    set_seed(2)
    act_func = 'elu'

    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(100, activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(data_shape,)
                )
            )

    model.add(Dense(2 , activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(100 , activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(data_shape,
                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse',optimizer='adam')

    return model

def ae_anomaly(load, data_size, validate=False, graph=None):
    weight_file_path = r'.\trained-model\ae_classifier.h5'
    #weight_file_path = r'C:\Users\bryan\Desktop\ae_classifier.h5'
    # Train model for 50 epochs, batch size of 10: 
    NUM_EPOCHS=20
    BATCH_SIZE=64
    label = 'malware_label'
    extreme = False

    if data_size == 'big':
       csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'
    elif data_size == 'small':
        csv_data_file = r'test-train-data\test\test_train_data-all.csv'

    #big_csv = r'test-train-data\validate\validate_data_new-all.csv'
    #small_csv = r'test-train-data\test\test_train_data-all.csv'
    #small = pd.read_csv(small_csv)
    #big = pd.read_csv(big_csv)
    #full_dataset = pd.concat([small, big], axis=0)
    #full_dataset = full_dataset.sample(n=data_size, random_state=1)
    #full_dataset = full_dataset.reset_index()
    #full_dataset = full_dataset.drop(['index'], axis=1)
    #print(full_dataset)
    full_dataset = pd.read_csv(csv_data_file)
    label_data = full_dataset.malware_label

    if validate:
        full_dataset = full_dataset.drop([label], axis=1)
        data_to_test = pd.DataFrame(MinMaxScaler().fit_transform(full_dataset),
                                columns=full_dataset.columns,
                                index=full_dataset.index)

        model = create_model(data_to_test.shape[1])
        model.load_weights(weight_file_path)

        predictions = model.predict(np.array(data_to_test))
        predictions = pd.DataFrame(predictions, columns=data_to_test.columns)
        predictions.index = data_to_test.index
        threshold = AE_threshold(data_to_test, predictions, extreme)
        scored = pd.DataFrame(index=data_to_test.index)
        scored['Loss_mae'] = np.mean(np.abs(predictions-data_to_test), axis=1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    else:
        x_train = full_dataset[full_dataset.malware_label == 0]
        x_train = x_train.drop(label, axis=1)
        x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train),
                                columns=x_train.columns,
                                index=x_train.index)
        x_test = full_dataset[full_dataset.malware_label == 1]
        x_test = x_test.drop(label, axis=1)
        x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test),
                                columns=x_test.columns,
                                index=x_test.index)

        model = create_model(x_train.shape[1])
        
        if load == 'train_and_save':
            write_model = ModelCheckpoint(filepath=weight_file_path,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        verbose=0)
            history = model.fit(np.array(x_train),np.array(x_train),
                            batch_size=BATCH_SIZE, 
                            epochs=NUM_EPOCHS,
                            validation_split=0.05,
                            verbose = 1,
                            callbacks = [write_model]).history
        elif load == 'train_only':
            history = model.fit(np.array(x_train),np.array(x_train),
                            batch_size=BATCH_SIZE, 
                            epochs=NUM_EPOCHS,
                            validation_split=0.05,
                            verbose = 1)

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
    
    print('Accuracy: {}'.format(accuracy_score(label_data, scored['Anomaly'])))
    print('Precision: {}'.format(precision_score(label_data, scored['Anomaly'])))
    print('Recall: {}'.format(recall_score(label_data, scored['Anomaly'])))
    print('F2 Score: {}'.format(fbeta_score(label_data, scored['Anomaly'], beta=2.0)))
