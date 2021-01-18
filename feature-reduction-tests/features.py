import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pingouin as pg
import matplotlib.patches as mpatches

from scipy.stats import bartlett, levene
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.random import set_seed
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier


def malware_distribution(data, label):
    '''
    Generate a bar chart showing malware to benign data ratio
    '''
    # Specify data column to calculate
    target = data[label]
    # Get dataset length for percentage calculation
    total = len(data)
    # Define graph area and title
    plt.figure(figsize = (6, 6))
    plt.title("Malware Dataset Distribution")

    # Generate count plot and turn into bar graph for display
    ax = sns.countplot(target)
    for p in ax.patches:
        percentage = '{:.0f}%'.format(p.get_height() / total * 100)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 5
        ax.annotate(percentage, (x, y), ha = 'center')

    plt.show()

def dataset_heatmap (data, label, annotate=False):
    '''
    Analyze all features together in a large heatmap. This is nearly impossible to interpret using all 519 features.
    '''
    #Gausidan distrabution of dataset
    data_features = data.drop(label, axis=1)
    data_standard_dev = (data_features - data_features.mean()) / data_features.std()
    gaussian_data = pd.concat([data[label], data_standard_dev], axis=1)

    # Define plot area
    plt.figure(figsize = (10, 8))
    plt.title("Correlation Heatmap")
    correlation = gaussian_data.corr()
    if annotate:
        sns.heatmap(correlation, annot = annotate, fmt = '.2f', cmap = 'coolwarm')
    else:
        sns.heatmap(correlation, annot = annotate, cmap = 'coolwarm')
    
    plt.show()

def calculate_vif (data, label):
    '''
    Method to calculate Variance Inflation Factor (VIF) to determine multi-collinearity of data. Had difficulty
    interpreting this output...
    '''
    data = data.drop(label, axis=1)
    vif_data = pd.DataFrame()
    vif_data['feature'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]

    for feature in vif_data:
        print(feature)

def mal_ben_hist (data, label, graph_set, benign_percent):
    '''
    Generate a feature for feature histogram comparing the importance of various features in distinguishing between
    malware and benign traffic.
    '''
    malware_label = data.malware_label
    # Remove malware label
    data = data.drop(label, axis=1)
    #std_data = StandardScaler().fit_transform(data)
    #norm_data = normalize(data)
    mm_data = MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(mm_data, columns = data.columns)
    _, axes = plt.subplots(10, 3, figsize=(12, 9)) # 3 columns containing 10 figures

    begin = graph_set * 30
    end = begin + 30
    data_to_graph = data.iloc[:, begin:end]

    data_to_graph = pd.concat([data_to_graph, malware_label], axis=1)
    malware = data_to_graph[data_to_graph.malware_label == 1]
    benign = data_to_graph[data_to_graph.malware_label == 0]
    # Get a percentage of the benign data set to balance measurements for analysis
    # This can be changed to view graphs differently, but is very helpful to truly
    # see the differences between benign and malicious traffic side by side
    percent = int(len(benign) * (benign_percent / 100))
    benign = benign.sample(n=percent)
    ax = axes.ravel()
    for i in range(data_to_graph.shape[1] - 1):
        _, bins = np.histogram(data_to_graph.iloc[:, i], bins=40)
        ax[i].hist(malware.iloc[:, i], bins=bins, color='r', alpha=.5) # Red for malware
        ax[i].hist(benign.iloc[:, i], bins=bins, color='g', alpha=0.3) # Green for benign
        ax[i].set_title(data_to_graph.columns[i], fontsize=9)
        ax[i].axes.get_xaxis().set_visible(False) # Just want to see separation not measurements
        ax[i].set_yticks(())
    
    ax[0].legend(['malware', 'benign'], loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()


def calculate_pca (data, label, components=10, fit=False, graph=None):
    '''
    Used to generate a desired number of "Principle Components" from the input data and return the calculated
    output components as a pandas dataframe
    '''
    features = data.columns
    # Remove features from data
    data_vals = data.loc[:, features].values
    # Define the target/label
    data_label = data.loc[:, [label]].values
    # Normalize the dataset
    std_data = MinMaxScaler().fit_transform(data_vals)

    # Calculate the 10 most important components
    #pca = PCA(n_components = components)
    if fit:
        data_pca_vals = PCA().fit(std_data.data)
    else:
        data_pca_vals = PCA().fit_transform(std_data)
    #pca_dataframe = pd.DataFrame(data = data_pca_vals)
    #final_pca_dataframe = pd.concat([pca_dataframe, data[[label]]], axis=1)

    if graph == 'heatmap':
        correlation = final_pca_dataframe.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap = 'coolwarm')
        plt.show()
    elif graph == 'pairplot':
        sns.pairplot(final_pca_dataframe, kind='scatter', hue=label, markers=['o', 's'], palette='Set2')
        plt.show()
    elif graph == 'scatter':
        colors = {'0': 'darkblue', '1': 'darkorange'}
        plt.scatter(data_pca_vals[:, 0], data_pca_vals[:, 1],
                    c=pd.Series(data['malware_label']).astype(str).map(colors), edgecolor='none',
                    alpha=0.5, cmap='viridis')
        m = mpatches.Patch(color='darkblue', label='Benign')
        b = mpatches.Patch(color='darkorange', label='Malware')
        plt.legend(handles=[m, b])
        plt.show()
    elif graph == 'comp_curve':
        plt.plot(np.cumsum(data_pca_vals.explained_variance_ratio_))
        plt.xlabel('number of componsents')
        plt.ylabel('cumulative explained variance')
        plt.show()

    #return pd.DataFrame(final_pca_dataframe)

def random_forest(data, label, estimators, graph=None):
    '''
    Use a Random Forest to determine feature importance. Can also be used to return the top number of
    features determined by the "IF" statement of val_tuple[1] value. This is the threshold determined by
    the Random Forest regressor and can be determiend by generating and analyzing the bar chart.
    
    This proved to be between 65%-80% successful in classifing malicious traffic by itself, however, the SVM
    used in tls-mal-detect.py was more successful due to its robust ability to deal with outliers.
    ''' 
    seed(1)
    set_seed(2)
    SEED = 123
    DATA_SPLIT_PCT = 0.3
    data_label = data.malware_label
    data = data.drop(label, axis=1)
    features = data.columns

    # Standardize the dataset
    std_data = MinMaxScaler().fit_transform(data)

    x_train, x_test, train_labels, test_labels = train_test_split(std_data, data_label, random_state=SEED, test_size=DATA_SPLIT_PCT)

    regressor = RandomForestClassifier(n_estimators=estimators, random_state=SEED)
    regressor.fit(x_train, train_labels)
    predictions = regressor.predict(x_test)
    feat_series = regressor.feature_importances_

    if graph == 'bar':
        pd.Series(feat_series, index=features).nlargest(50).plot(kind='barh').invert_yaxis()
        plt.show()

    feature_list = [(feature, round(importance, 2)) for feature, importance in zip(list(features), list(feat_series))]
    top_10_feature_list = []
    for val_tuple in feature_list:
        if val_tuple[1] >= 0.03:
            top_10_feature_list.append(val_tuple[0])

    final_data = data[top_10_feature_list]
    final_data = pd.DataFrame(MinMaxScaler().fit_transform(final_data), columns=top_10_feature_list)
    final_data = pd.concat([data_label, final_data], axis=1)
    return final_data

def autoencoded_features (data, label, final_features, graph=None):
    '''
    Attempt to analyze and interpret data using a Sparse, Stacked Autoencoder. Was not too successful,
    but leaving here for potential, future analysis in feature reduction in lieu of Random Forest or
    PCA. Read some interesting research where that was successful.
    '''
    # Balance dataset based on percentage passed to function
    output_vals = []
    seed(1)
    set_seed(2)
    SEED = 123
    DATA_SPLIT_PCT = 0.2

    # Split into training and testing datasets
    x_train, x_test = train_test_split(data, test_size=DATA_SPLIT_PCT, random_state=SEED)
    x_train, x_valid = train_test_split(x_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
    
    x_train_0 = x_train.loc[data[label] == 0]
    x_train_1 = x_train.loc[data[label] == 1]
    x_train_0_x = x_train_0.drop([label], axis=1)
    x_train_1_x = x_train_1.drop([label], axis=1)

    x_valid_0 = x_valid.loc[data[label] == 0]
    x_valid_1 = x_valid.loc[data[label] == 1]
    x_valid_0_x = x_valid_0.drop([label], axis=1)
    x_valid_1_x = x_valid_1.drop([label], axis=1)

    x_test_0 = x_test.loc[data[label] == 0]
    x_test_1 = x_test.loc[data[label] == 1]
    x_test_0_x = x_test_0.drop([label], axis=1)
    x_test_1_x = x_test_1.drop([label], axis=1)

    scaler = StandardScaler().fit(x_train_0_x)
    x_train_0_x_rescaled = scaler.transform(x_train_0_x)
    x_valid_0_x_rescaled = scaler.transform(x_valid_0_x)
    x_valid_x_rescaled = scaler.transform(x_valid.drop([label], axis=1))

    x_test_0_x_rescaled = scaler.transform(x_test_0_x)
    x_test_x_rescaled = scaler.transform(x_test.drop([label], axis=1))

    # Autoencoder values
    learning_epochs = 200
    batch_size = 128
    input_dim = x_train_0_x_rescaled.shape[1]
    #input_dim = x_train_1.shape[1]
    encoding_dim = int(input_dim / 2)
    hidden_dim_1 = int(encoding_dim / 2)
    hidden_dim_2 = int(hidden_dim_1 / 2)
    final_hidden_dim = final_features
    learning_rate = 1e-6

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = Dense(final_hidden_dim, activation='relu')(encoder)
    decoder = Dense(final_hidden_dim, activation='relu')(encoder)
    decoder = Dense(hidden_dim_2, activation='relu')(decoder)
    decoder = Dense(hidden_dim_1, activation='relu')(decoder)
    decoder = Dense(encoding_dim, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
    write_model = ModelCheckpoint(filepath=r'C:\Users\bryan\Desktop\ae_data\model\ae_calssifier.h5', save_best_only=True, verbose=0)
    write_logs = TensorBoard(log_dir=r'C:\Users\bryan\Desktop\ae_data\logs', histogram_freq=0, write_graph=True, write_images=True)
    history = autoencoder.fit(x_train_0_x_rescaled, x_train_0_x_rescaled,
                            epochs=learning_epochs,
                            batch_size=batch_size,
                            validation_data=(x_valid_0_x_rescaled, x_valid_0_x_rescaled),
                            verbose=1,
                            callbacks=[write_model, write_logs]).history
    
    valid_x_predictions = autoencoder.predict(x_valid_x_rescaled)
    mse = np.mean(np.power(x_valid_x_rescaled - valid_x_predictions, 2), axis=1)

    error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': x_valid[label]})
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df['True_class'], error_df['Reconstruction_error'])
    threshold = np.mean(thresholds)
    threshold_fixed = float("{:0.4f}".format(threshold))
    roc_auc = auc(false_pos_rate, true_pos_rate,)
    
    output_vals.append('MSE: {}'.format(mse))
    output_vals.append('Threshold mean: {}'.format(threshold))
    output_vals.append('AUC: {}'.foramt(auc(false_pos_rate, true_pos_rate)))

    if graph == 'loss':
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    elif graph == 'pre_call':
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1:], label='Precision', linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label='Recall', linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()
    elif graph == 're_error':
        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='', label='Malware Estimation' if name == 1 else 'Benign Estimate')
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, label='Threshold')
        ax.legend()
        plt.title('Reconstruction error for malicious/benign traffic')
        plt.ylabel('Reconstruction error')
        plt.xlabel('Data point index')
        plt.show()
    elif graph == 'heatmap':
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]
        conf_matrix = confusion_matrix(error_df['True_class'], pred_y)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix,
                    xticklabels=['Benign', 'Malware'],
                    yticklabels=['Benign', 'Malware'],
                    annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
    elif graph == 'roc':
        plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
        plt.plot([0,1],[0,1], linewidth=5)
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Reciever operating charactistic curve (ROC)')
        plt.ylabel('True Positive Rating')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    return output_vals

def factor_analysis(data, label, eq_var=True):
    '''
    Method to perform factor analysis to determine the most important features and calculate
    the ability to distinguish between malware and benign traffic using varying numbers of "important"
    features.
    '''
    np.seterr(divide='raise')
    for feature in data.columns:
        try:
            bart_vals = pg.homoscedasticity(data, dv=feature, group='malware_label', method='bartlett')
            lev_vals = pg.homoscedasticity(data, dv=feature, group='malware_label')
            if bart_vals['equal_var'].values[0] == eq_var:
                print("Bartlett Value feature: {}\nT Value: {}\nP Value: {}".format(feature, bart_vals['T'].values[0], bart_vals['pval'].values[0]))
            elif lev_vals['equal_var'].values[0] == eq_var:
                print("Levene Value feature: {}\nT Value: {}\nP Value: {}".format(feature, bart_vals['T'].values[0], bart_vals['pval'].values[0]))
        except Exception as e:
            pass

#if __name__ == '__main__':
    #def load_input (csv_data_file):
csv_data_file = r'test-train-data\test\test_train_data-all.csv'
csv_data_file_new = r'test-train-data\validate\validate_data_new-all.csv'
dataset = pd.read_csv(csv_data_file)
new_dataset = pd.read_csv(csv_data_file_new)

full_data = pd.concat([dataset, new_dataset], axis=0)
full_data = full_data.reset_index()
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