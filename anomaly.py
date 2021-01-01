import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import ModelCheckpoint

def cov_matrix_func(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def split_data(label):
    csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'
    dataset = pd.read_csv(csv_data_file)

    x_train = dataset[dataset.malware_label == 0]
    x_train = x_train.drop(label, axis=1)
    x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train))
    x_test = dataset[dataset.malware_label == 1]
    x_test = x_test.drop(label, axis=1)
    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test))

    return x_train, x_test

def create_model(data_shape):
    seed(1)
    set_seed(2)
    act_func = 'elu'

    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(10 , activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(data_shape,)
                )
            )

    model.add(Dense(2 , activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(10 , activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(data_shape,
                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse',optimizer='adam')

    return model

def ae_anomaly(label, load=False, write=False, graph=None):
    weight_file_path = r'C:\Users\bryan\Desktop\ae_calssifier.h5'
    x_train, x_test = split_data(label)
    model = create_model(x_train.shape[1])

    if load:
        model.load_weights(weight_file_path)
    else:
        # Train model for 100 epochs, batch size of 10: 
        NUM_EPOCHS=50
        BATCH_SIZE=10
        write_model = ModelCheckpoint(filepath=weight_file_path,
                                    save_best_only=False,
                                    save_weights_only=True,
                                    verbose=0)
        history = model.fit(np.array(x_train),np.array(x_train),
                        batch_size=BATCH_SIZE, 
                        epochs=NUM_EPOCHS,
                        validation_split=0.05,
                        verbose = 1,
                        callbacks = [write_model]).history

    X_pred = model.predict(np.array(x_test))
    X_pred = pd.DataFrame(X_pred, 
                        columns=x_test.columns)
    X_pred.index = x_test.index

    scored = pd.DataFrame(index=x_test.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-x_test), axis = 1)
    scored['Threshold'] = 0.016
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    scored.head()

    X_pred_train = model.predict(np.array(x_train))
    X_pred_train = pd.DataFrame(X_pred_train, 
                        columns=x_train.columns)
    X_pred_train.index = x_train.index

    scored_train = pd.DataFrame(index=x_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-x_train), axis = 1)
    scored_train['Threshold'] = 0.016
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])
    
    if write:
        scored.to_csv(r'C:\Users\bryan\Desktop\AE_Anomaly_distance.csv')

    if graph == 'loss':
        plt.plot(history.history['loss'],
                'b',
                label='Training loss')
        plt.plot(history.history['val_loss'],
                'r',
                label='Validation loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss, [mse]')
        plt.ylim([0,.02])
        plt.show()
    elif graph == 'md':
        X_pred = model.predict(np.array(x_train))
        X_pred = pd.DataFrame(X_pred, 
                            columns=x_train.columns)
        X_pred.index = x_train.index

        scored = pd.DataFrame(index=x_train.index)
        scored['Loss_mae'] = np.mean(np.abs(X_pred-x_train), axis = 1)
        plt.figure()
        sns.distplot(scored['Loss_mae'],
                    bins = 10, 
                    kde= True,
                    color = 'blue')
        plt.xlim([0.0,.02])
        plt.show()
    elif graph == 'thresh':
        scored.plot(logy=True,  figsize = (10,6), ylim = [1e-3,1e0], color = ['blue','red'])
        plt.show()
