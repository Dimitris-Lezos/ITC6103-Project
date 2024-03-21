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

data = pd.read_csv('C:/Users/dlezo/OneDrive - The American College of Greece/ITC6103 Applied Machine Learning/Project/train.csv')
data.head()
features = pd.DataFrame({
    'Count': data.count(),
    'Unique': data.nunique(),
    'Unique Percent': round(((data.nunique()/ data.count()) * 100),2),
    'Null Count': data.isnull().sum(),
    'Null Percent': data.isna().sum()* 100 / data.shape[0],
    'Data Type': data.dtypes
})
data['action_type'].unique()
data['platform'].unique()
data['city'].nunique()
plt.figure(figsize = (15, 5))
sns.countplot(data=data, x='platform', order = data['platform'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Frequency per Platform', fontsize = 15)
plt.xlabel('Class', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
data['device'].unique()
impressions_filter = data[data['impressions'].notna()]
impressions_filter['action_type'].unique()

prices_filter = data[data['prices'].notna()]
prices_filter['action_type'].unique()

data = data.loc[(data['platform'] == 'BR')]
data.shape
# subset_df = original_data.loc[(original_data['platform'] == 'UK')]
# # subset_df = original_data.loc[(original_data['platform'] == 'BR') | (original_data['platform'] == 'US')]
# subset_df.head()

# subset_df.shape




# subset_df[subset_df['current_filters'].notna()]
# subset_df.loc[subset_df['current_filters'] != '', 'current_filters'].unique()
# unique_filters = set()

# for item in subset_df['current_filters']:
#     if pd.notna(item):
#         for value in item.split('|'):
#             unique_filters.add(value)
# unique_filters = list(unique_filters)

# print(unique_filters)
# len(unique_filters)
# expanded_filters_df = pd.DataFrame(index = subset_df.index, columns = unique_filters)

# expanded_filters_df = expanded_filters_df.fillna(0)

# update_values = lambda row: [expanded_filters_df.at[row.name, value] if isinstance(row['current_filters'], str) and value in row['current_filters'].split('|') else 0 for value in expanded_filters_df.columns]

# # Apply the lambda function row-wise to subset_df
# expanded_filters_df = subset_df.apply(update_values, axis=1, result_type='expand')

# # for value in unique_filters:
# #     expanded_filters_df[value] = 0
# #     expanded_filters_df.loc[subset_df['current_filters'].apply(lambda x: value in x if isinstance(x, list) else False), value] = 1

# expanded_filters_df.head()
# subset_df = subset_df.join(expanded_filters_df)
# subset_df.drop(columns=['current_filters'], inplace=True)
# subset_df[subset_df['current_filters'].notna()]
data.head(5)


## Starting with our classes, the request here is whether there was at least one clickout item or not.
## So we assign value 1 if the clickout items were one or more & 0 if there were no clickouit items.

def clickout_item_counts(data):
    clickout_item_count = (data['action_type'] == 'clickout item').sum()

    return 1 if clickout_item_count >= 1 else 0


## Checking if for every session_id we always have a single platform
data.groupby('session_id')['platform'].nunique().sort_values(ascending=False).head()
## Hypothesis true

## Will extract single platform for each session_id
def platform(data):
    platform = data['platform'].unique()[0]
    return platform
## Checking if for every session_id we always have a single device
data.groupby('session_id')['device'].nunique().sort_values(ascending = False).head()
## Hypothesis true


## Will extract single device for each session_id
def device(data):
    device = data['device'].unique()[0]
    return device
## Checking if for every session_id we always have a single city
data.groupby('session_id')['city'].nunique().sort_values(ascending = False).head()
## Hypothesis false





## Checking how many cities exist in the dataset
data['city'].nunique()


## For cities since we have a huge count (even solely for BR platform) we will not create individual features for each city
## Instead we are keeping a metric for how many cities the user searched for in a specific session
def city_count(data):
    city_count = data['city'].nunique()

    return city_count


## Calculating sessions duration - since it's interval data we can take the substraction without converting

def session_duration(data):
    first_step_timestamp = data['timestamp'].min()
    last_step_timestamp = data['timestamp'].max()
    session_duration = last_step_timestamp - first_step_timestamp

    return session_duration


## Calculating total session steps

def steps_count(data):
    steps_count = data['step'].max()

    return steps_count


def interaction_item_rating_count(data):
    interaction_item_rating_count = (data['action_type'] == 'interaction item rating').sum()

    return interaction_item_rating_count




def interaction_item_info_count(data):
    interaction_item_info_count = (data['action_type'] == 'interaction item info').sum()

    return interaction_item_info_count


def interaction_item_image_count(data):
    interaction_item_image_count = (data['action_type'] == 'interaction item image').sum()

    return interaction_item_image_count


def interaction_item_deals_count(data):
    interaction_item_deals_count = (data['action_type'] == 'interaction item deals').sum()

    return interaction_item_deals_count


def change_of_sort_order_count(data):
    change_of_sort_order_count = (data['action_type'] == 'change of sort order').sum()

    return change_of_sort_order_count


def filter_selection_count(data):
    filter_selection_count = (data['action_type'] == 'filter selection').sum()

    return filter_selection_count


def search_for_item_count(data):
    search_for_item_count = (data['action_type'] == 'search for item').sum()

    return search_for_item_count



def search_for_destination_count(data):
    search_for_destination_count = (data['action_type'] == 'search for destination').sum()

    return search_for_destination_count



def search_for_poi_count(data):
    search_for_poi_count = (data['action_type'] == 'search for poi').sum()

    return search_for_poi_count



## Simply calculating if a user used a filter within a specific session or not - trying to find another way but I either don't have the memory or it's not working
## Actually not keeping even this since we have action_type filter selection

def filters_used(data):
    if data['current_filters'].notna().any():
        return 1
    else:
        return 0


def transformation(data):
    metrics = {
        'Clickout_Item_Counts': clickout_item_counts(data),
        'Platform': platform(data),
        'Device': device(data),
        'City_Count': city_count(data),
        'Session_Duration': session_duration(data),
        'Steps_Count': steps_count(data),
        'Interaction_Item_Rating_Count': interaction_item_rating_count(data),
        'Interaction_Item_Image_Count': interaction_item_image_count(data),
        'Interaction_Item_Deals_Count': interaction_item_deals_count(data),
        'Change_of_Sort_Order_Count': change_of_sort_order_count(data),
        'Filter_Selection_Count': filter_selection_count(data),
        'Search_for_Item_Count': search_for_item_count(data),
        'Search_for_Destination_Count': search_for_destination_count(data),
        'Search_for_Poi_Count': search_for_poi_count(data)
    }
    return pd.Series(metrics)

new_df = data.groupby(['user_id', 'session_id']).apply(transformation)
new_df.head()

new_df.shape




features = pd.DataFrame({
    'Count': new_df.count(),
    'Unique': new_df.nunique(),
    'Unique Percent': round(((new_df.nunique()/ new_df.count()) * 100),2),
    'Null Count': new_df.isnull().sum(),
    'Null Percent': new_df.isna().sum()* 100 / new_df.shape[0],
    'Data Type': new_df.dtypes
})

new_df.describe()



## Checking for duplicates
print(new_df.shape)
df_drop_dup = new_df.drop_duplicates()
print(df_drop_dup.shape)
## No duplicates



new_df['Clickout_Item_Counts'].value_counts()

sns.countplot(data = new_df, x = 'Clickout_Item_Counts')
plt.title('Frequency per Class', fontsize = 15)
plt.xlabel('Class', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)


class1 = new_df[new_df['Clickout_Item_Counts'] == 1]
class0 = new_df[new_df['Clickout_Item_Counts'] == 0]
print(class1.shape)
print(class0.shape)

sample_class1 = resample(
    class1,
    n_samples = 5000,
    replace = False,
    random_state = 1
    )
sample_class0 = resample(
    class0,
    n_samples = 5000,
    replace = False,
    random_state = 1
    )
balanced_df = pd.concat([sample_class1, sample_class0])

balanced_df['Clickout_Item_Counts'].value_counts()

sns.countplot(data = balanced_df, x = 'Clickout_Item_Counts')
plt.title('Frequency per Class', fontsize = 15)
plt.xlabel('Class', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)


# corr_matrix = balanced_df.corr()
#
# f, ax = plt.subplots(figsize=(14, 8))
# sns.heatmap(corr_matrix,
#             annot=True,
#             annot_kws={'size': 8},
#             cmap="Spectral_r");

features = pd.DataFrame({
    'Count': balanced_df.count(),
    'Unique': balanced_df.nunique(),
    'Unique Percent': round(((balanced_df.nunique()/ balanced_df.count()) * 100),2),
    'Null Count': balanced_df.isnull().sum(),
    'Null Percent': balanced_df.isna().sum()* 100 / balanced_df.shape[0],
    'Data Type': balanced_df.dtypes
})

balanced_df.to_csv('./mini.csv')


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

plt.figure(figsize=(18,10))
ax = sns.boxplot(data=X_train_numerical)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show();






scaler = MinMaxScaler()
X_train_numerical = pd.DataFrame(scaler.fit_transform(X_train_numerical), index = X_train_numerical.index, columns = X_train_numerical.columns)
X_test_numerical  = pd.DataFrame(scaler.transform(X_test_numerical), index = X_test_numerical.index, columns = X_test_numerical.columns)

plt.figure(figsize = (18, 10))
ax = sns.boxplot(data = X_train_numerical)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.show();

plt.figure(figsize = (18, 10))
ax = sns.boxplot(data = X_test_numerical)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.show();

X_train_categorical = pd.get_dummies(X_train_categorical)
X_test_categorical = pd.get_dummies(X_test_categorical)
print(X_train_categorical.head(2))
print(X_test_categorical.head(2))

X_train = X_train_categorical.join(X_train_numerical)
X_test = X_test_categorical.join(X_test_numerical)
print('Shape of x_train and y_train: ', X_train.shape, y_train.shape)
print('Shape of x_test and y_test: ',   X_test.shape,  y_test.shape)

###########################################################################
clf_results = pd.DataFrame([])




parameter_space_NN = {
    'hidden_layer_sizes': [(5,), (10,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'batch_size': [10 , 20],
    'tol': [1e-7],
    'learning_rate': ['constant','adaptive'],
}
clf_NN = MLPClassifier(random_state = 1,
                       max_iter = 5000,
                       verbose = False,
                       validation_fraction = 0.1
                      )
NN = GridSearchCV(estimator = clf_NN,
                  param_grid = parameter_space_NN,
                  n_jobs = -1,
                  cv = 5)
NN.fit(X_train, y_train)

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











df = pd.DataFrame()
df.to_csv()



































