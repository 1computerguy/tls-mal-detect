import pandas as pd

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier

def pca_output (data, label, components=10):
    '''
    Used to generate a desired number of "Principle Components" from the input data and return the calculated
    output components as a pandas dataframe
    '''
    data_label = data[[label]]
    data = data.drop([label], axis=1)
    std_data = MinMaxScaler().fit_transform(data.values)

    # Calculate the 10 most important components
    pca = PCA(n_components = components)
    data_pca_vals = pca.fit_transform(std_data)
    pca_dataframe = pd.DataFrame(data = data_pca_vals)
    final_pca_dataframe = pd.concat([pca_dataframe, data_label], axis=1)

    return pd.DataFrame(final_pca_dataframe)


def random_forest(data, label, estimators):
    '''
    Use a Random Forest to determine feature importance. Can also be used to return the top number of
    features determined by the "IF" statement of val_tuple[1] value. This is the threshold determined by
    the Random Forest regressor and can be determiend by generating and analyzing the bar chart.
    
    This proved to be between 65%-80% successful in classifing malicious traffic by itself, however, the SVM
    used in svm-testing.py was more successful due to its robust ability to deal with outliers.
    ''' 
    seed(1)
    set_seed(2)
    SEED = 123
    data_label = data.malware_label
    data = data.drop([label], axis=1)
    features = data.columns

    # Standardize the dataset
    std_data = MinMaxScaler().fit_transform(data)

    regressor = RandomForestClassifier(n_estimators=estimators, random_state=SEED)
    regressor.fit(std_data, data_label)
    feat_series = regressor.feature_importances_

    feature_list = [(feature, round(importance, 2)) for feature, importance in zip(list(features), list(feat_series))]
    top_x_feature_list = []
    for val_tuple in feature_list:
        if val_tuple[1] >= 0.03:
            top_x_feature_list.append(val_tuple[0])

    final_data = pd.DataFrame(MinMaxScaler().fit_transform(data[top_x_feature_list]))
    final_data = pd.concat([final_data, data_label], axis=1)

    # Return dataframe of N most important features based on feature_list importances and 
    # val_tuple[0] measure value
    return final_data

def autoencoded_features (data, label, feature_count):
    '''
    Attempt to analyze and interpret data using a Sparse, Stacked Autoencoder. Was not too successful,
    but leaving here for potential, future analysis in feature reduction in lieu of Random Forest or
    PCA. Read some interesting research where that was successful.
    '''
    # Balance dataset based on percentage passed to function
    seed(1)
    set_seed(2)
    data_label = data.malware_label
    data = data.drop([label], axis=1)
    std_data = MinMaxScaler().fit_transform(data)

    # Autoencoder values
    input_dim = data.shape[1]
    encoding_dim = int(input_dim / 2)
    hidden_dim_1 = int(encoding_dim / 2)
    hidden_dim_2 = int(hidden_dim_1 / 2)
    final_hidden_dim = feature_count
    learning_rate = 1e-6

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = Dense(final_hidden_dim, activation='relu')(encoder)
    
    autoencoder_model = Model(inputs=input_layer, outputs=encoder)
    autoencoder_feature_output = pd.DataFrame(autoencoder_model.predict(std_data))
    autoencoder_feature_output = pd.concat([autoencoder_feature_output, data_label], axis=1)
    
    return autoencoder_feature_output

if __name__ == '__main__':
    #def load_input (csv_data_file):
    csv_data_file = 'test_train_data-all.csv'
    dataset = pd.read_csv(csv_data_file)
    # Remove columns filled with all 0 value (these will be statistically insignifant and will cause
    # issues when using correlation methods of analysis)
    data_no_z_cols = dataset.loc[:, (dataset != 0).any(axis=0)]

    # NOTE TO SELF
    # To import as module for testing:
    # from importlib import import_module, reload
    # a = import_module('features')
    # Use this when you make changes
    # reload(a)
    # 
    # Call methods using below syntax:
    #  a.random_forest(a.dataset, 'labelname', 100, 'bar')