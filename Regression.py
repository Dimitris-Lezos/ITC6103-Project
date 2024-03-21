from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


###########################################################
data = pd.read_csv('cubic_zirconia.csv')
data.replace('Fair', 0, inplace=True)
data.replace('Good', 1, inplace=True)
data.replace('Very Good', 2, inplace=True)
data.replace('Premium', 3, inplace=True)
data.replace('Ideal', 4, inplace=True)
data.replace('FL', 10, inplace=True)
data.replace('IF', 9, inplace=True)
data.replace('VVS1', 8, inplace=True)
data.replace('VVS2', 7, inplace=True)
data.replace('VS1', 6, inplace=True)
data.replace('VS2', 5, inplace=True)
data.replace('SI1', 4, inplace=True)
data.replace('SI2', 3, inplace=True)
data.replace('I1', 2, inplace=True)
data.replace('I2', 1, inplace=True)
data.replace('I3', 0, inplace=True)
data.replace('D', 6, inplace=True)
data.replace('E', 5, inplace=True)
data.replace('F', 4, inplace=True)
data.replace('G', 3, inplace=True)
data.replace('H', 2, inplace=True)
data.replace('I', 1, inplace=True)
data.replace('J', 0, inplace=True)
#data.drop(['x', 'y', 'z', 'carat'], axis='columns',inplace=True)
############################################################
data_concrete = pd.read_csv('Concrete_Data_Yeh.csv')
data_concrete['coarse_vs_fine'] = data_concrete['coarseaggregate']/data_concrete['fineaggregate']
data_concrete['coarse_and_fine'] = data_concrete['coarseaggregate']+data_concrete['fineaggregate']
############################################################
data_honey = pd.read_csv('honey_purity_dataset.csv')
data_honey.drop(['Pollen_analysis'], axis='columns',inplace=True)
data_honey.drop(['Viscosity'], axis='columns',inplace=True)
#data_honey.drop(['Price'], axis='columns',inplace=True)
############################################################
#for data in [data_honey]: #[data, data_concrete, data_honey]:
for data in [data, data_concrete, data_honey]:
    #diabetes_X = diabetes.data[:, featureNumber].reshape(-1,1)
    data_normalized = MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(data_normalized, columns=data.columns)
    data.head()
    data.shape
    features = pd.DataFrame({
        'Count': data.count(),
        'Unique': data.nunique(),
        'Unique Percent': round(((data.nunique()/ data.count()) * 100),2),
        'Null Count': data.isnull().sum(),
        'Null Percent': data.isna().sum()* 100 / data.shape[0],
        'Data Type': data.dtypes
    })
    print(features)
    print(data.describe())
    corr_matrix = data.corr()
    f, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                annot_kws={'size': 8},
                cmap="Spectral_r")
    plt.show()
    plt.figure(figsize=(18, 10))
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.show()