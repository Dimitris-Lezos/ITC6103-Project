import time

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

pd.set_option('display.max_columns', 100)
pd.set_option("display.precision", 2)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 300)

balanced_df = pd.read_csv('./mini.csv')
balanced_df.groupby(['user_id', 'session_id'])

X = balanced_df.drop('Clickout_Item_Counts', axis=1)
y = balanced_df['Clickout_Item_Counts']
print ("X dimensions: ", X.shape)
print ("y dimensions: ", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.30,
                                                    stratify = y,
                                                    random_state = 1)
print('Shape of x_train and y_train: ', X_train.shape, y_train.shape)
print('Shape of x_test and y_test: ',   X_test.shape,  y_test.shape)

X_train_numerical = X_train.select_dtypes(include = np.number)
X_train_categorical = X_train.select_dtypes(include = 'object')
X_test_numerical = X_test.select_dtypes(include = np.number)
X_test_categorical = X_test.select_dtypes(include = 'object')

print(X_train_numerical.columns)
print(X_train_categorical.columns)
print(X_test_numerical.columns)
print(X_test_categorical.columns)

# plt.figure(figsize=(18,10))
# ax = sns.boxplot(data=X_train_numerical)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
# plt.show()

scaler = MinMaxScaler()
X_train_numerical = pd.DataFrame(scaler.fit_transform(X_train_numerical), index = X_train_numerical.index, columns = X_train_numerical.columns)
X_test_numerical  = pd.DataFrame(scaler.transform(X_test_numerical), index = X_test_numerical.index, columns = X_test_numerical.columns)

# plt.figure(figsize = (18, 10))
# ax = sns.boxplot(data = X_train_numerical)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
# plt.show()

# plt.figure(figsize = (18, 10))
# ax = sns.boxplot(data = X_test_numerical)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
# plt.show()

X_train_categorical = X_train_categorical.drop(['user_id','session_id'], axis='columns')
X_test_categorical = X_test_categorical.drop(['user_id','session_id'], axis='columns')

X_train_categorical = pd.get_dummies(X_train_categorical)
X_test_categorical = pd.get_dummies(X_test_categorical)
print(X_train_categorical.head(2))
print(X_test_categorical.head(2))

X_train = X_train_categorical.join(X_train_numerical)
X_test = X_test_categorical.join(X_test_numerical)
print('Shape of x_train and y_train: ', X_train.shape, y_train.shape)
print('Shape of x_test and y_test: ',   X_test.shape,  y_test.shape)

###########################################################################

X_train = np.array(X_train)
X_test = np.array(X_test)


clf_results = pd.DataFrame([])

parameter_space_NN = {
    'hidden_layer_sizes': [(5,), (10,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'batch_size': [10 , 20],
    'batch_size': [100],
    'tol': [1e-7],
    'learning_rate': ['constant','adaptive'],
}
clf_NN = MLPClassifier(random_state = 1,
                       max_iter = 5000,
                       verbose = False,
                       validation_fraction = 0.1,
                      )
NN = GridSearchCV(estimator = clf_NN,
                  param_grid = parameter_space_NN,
                  n_jobs = -1,
                  cv = 5,
                  )
start_time = time.time()
NN.fit(X_train, y_train)
print("##### Computation time: ", time.time()-start_time, "seconds")

print("Best score: ", NN.best_score_)
print("Best parameters: ", NN.best_params_)
print("Best estimator: ", NN.best_estimator_)

y_train_pred_NN = NN.predict (X_train)
y_test_pred_NN = NN.predict (X_test)
clf_results = pd.concat([clf_results, pd.json_normalize({'Model': 'Neural Network',
                                                         'Accuracy': round(metrics.accuracy_score(y_test, y_test_pred_NN), 3),
                                                         'F1_macro': round(metrics.f1_score(y_test, y_test_pred_NN, average='macro'), 3)})])
clf_results

# Neural Network

confMatrixTrainNN = confusion_matrix(y_train, y_train_pred_NN, labels = None)
confMatrixTestNN = confusion_matrix(y_test, y_test_pred_NN, labels = None)

print ('Neural Network: Confusion matrix Train Set')
print (confMatrixTrainNN)
print ()

print ('Neural Network: Confusion matrix Test Set')
print (confMatrixTestNN)
print ()

ConfusionMatrixDisplay(confusion_matrix=confMatrixTestNN).plot()
plt.title('Neural Network Confusion Matrix')
plt.show()

# Neural Network

print ('Neural Network: Classification Report')
print(classification_report(y_test, y_test_pred_NN))


# Neural Network

# First we compute the probability for the classes (test set)
pr_y_test_pred_NN = NN.predict_proba(X_test)
fprNN, tprNN, thresholdsNN = roc_curve(y_test, pr_y_test_pred_NN[:,1])
roc_curve(y_test, pr_y_test_pred_NN[:,1])

lw=2
plt.plot(fprNN, tprNN, color = 'blue', label = 'Neural Network')
# plt.plot(fprDT, tprDT, color = 'red', label = 'Decision Tree')
# plt.plot(fprKNN, tprKNN, color = 'green', label = 'K-Nearest Neighbor')
# plt.plot(fprNB, tprNB, color = 'yellow', label = 'Naive Bayes')
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve for the recurring cancer')
plt.legend(loc = "lower right")
plt.show()
print ('AUC_NN = ',metrics.auc(fprNN, tprNN))
# print ('AUC_DT = ',metrics.auc(fprDT, tprDT))
# print ('AUC_KNN = ',metrics.auc(fprKNN, tprKNN))
# print ('AUC_NB = ',metrics.auc(fprNB, tprNB))









































