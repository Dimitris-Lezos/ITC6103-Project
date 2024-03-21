from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


###########################################################
data = pd.read_csv('train.csv')
#################################################################################
# Preprocess data:
#### 1. Check for missing values
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
#### 2. Check for duplicates
duplicates = data.duplicated()
print(duplicates.loc[duplicates == True])
# No Duplicates
#### 3. Encode Ordinal features
#cuts = [['Fair'], ['Good'], ['Very Good'], ['Premium'], ['Ideal'], ['Fair']]
## 3.a Ordinal Encoding
cuts = data[['cut', 'color', 'clarity']]
# Worst -> Best: 0, 1, 2...
encoder = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                     ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                     ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']])
# Fit and transform the data
data[['cut', 'color', 'clarity']] = encoder.fit_transform(cuts)
## 3.b Helmert Encoding
## Very poor results
# # Instantiate the encoder
# encoder = ce.HelmertEncoder(cols=['cut', 'color', 'clarity'])
# # Fit and transform the data
# df_encoded = encoder.fit_transform(data[['cut', 'color', 'clarity']])
# price = data[['price']]
# data.drop(columns=['price'], inplace=True)
# data = data.join(df_encoded)
# data = data.join(price)
# data.drop(columns=['cut', 'intercept', 'id', 'color', 'clarity'], inplace=True)
## Print the encoded data
print(data.head())
#### 3.5 Remove the id column
data.drop(columns=['id'], inplace=True)
#### 4. Normalize the data in the numerical columns?
# price = data[['price']]
# data = pd.DataFrame(MinMaxScaler().fit_transform(data.iloc[:,:-1]), columns=data.columns[:-1])
# data = data.join(price)
# try:
#     data.drop(columns=['id'], inplace=True)
# except:
#     pass
#### 4. Split between training and test data
# Add price / carat
data['price_carat'] = data['price'] # / data['carat']
x_train, x_test, y_train, y_test = train_test_split (data.iloc[:,:-2],data.iloc[:,-2:], test_size=0.3, random_state=42)
# Verify that the split gives similar distributions between train and test on the Ordinal columns
try:
    print(x_train['cut'].value_counts(normalize=True), x_test['cut'].value_counts(normalize=True))
    print(x_train['color'].value_counts(normalize=True), x_test['color'].value_counts(normalize=True))
    print(x_train['clarity'].value_counts(normalize=True), x_test['clarity'].value_counts(normalize=True))
except:
    pass
# Also show in plots
# plt.figure(1)
# plt.plot(x_train['cut'].value_counts(normalize=True, sort=False).sort_index()+0.02)
# plt.plot(x_test['cut'].value_counts(normalize=True, sort=False).sort_index())
# plt.figure(2)
# plt.plot(x_train['color'].value_counts(normalize=True, sort=False).sort_index()+0.02)
# plt.plot(x_test['color'].value_counts(normalize=True, sort=False).sort_index())
# plt.figure(3)
# plt.plot(x_train['clarity'].value_counts(normalize=True, sort=False).sort_index()+0.02)
# plt.plot(x_test['clarity'].value_counts(normalize=True, sort=False).sort_index())
# plt.show()
#### 5. Investigate the correlation of the data
# print(data.describe())
# corr_matrix = data.corr()
# f, ax = plt.subplots(figsize=(14, 8))
# sns.heatmap(corr_matrix,
#             annot=True,
#             annot_kws={'size': 8},
#             cmap="Spectral_r")
# plt.show()
# We see that carat, x, y, z are completely correlated, so we remove x, y & z features
data.drop(columns=['x', 'y', 'z'], inplace=True)
x_train.drop(columns=['x', 'y', 'z'], inplace=True)
x_test.drop(columns=['x', 'y', 'z'], inplace=True)
# # Let's see the correlation again
# corr_matrix = data.corr()
# f, ax = plt.subplots(figsize=(14, 8))
# sns.heatmap(corr_matrix,
#             annot=True,
#             annot_kws={'size': 8},
#             cmap="Spectral_r")
# plt.show()
# Price and carat are very closely correlated as expected we will try to train for two cases:
# a. Price
# b. Price / carat
#### 5.5 Boxplots
# plt.figure(4)
# plot = sns.boxplot(data=data.drop(columns=['id', 'price', 'price_carat']))
# plt.show()
#### 6. Do a simple linear regression for each column
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanSquaredError


learning_rate = 0.01
# Creating model using the Sequential in tensorflow
model = Sequential([
Dense(100, kernel_initializer='normal', activation='relu'),
Dropout(0.2),
Dense(200, kernel_initializer='normal', activation='relu'),
Dropout(0.2),
Dense(50, kernel_initializer='normal', activation='relu'),
Dense(1, kernel_initializer='normal', activation='linear')
])

# loss function
msle = MeanSquaredLogarithmicError()
mse = MeanSquaredError()
model.compile(
    loss=msle,
    optimizer=Adam(learning_rate=learning_rate),
    #metrics=[msle],
    metrics = [mse],
)

# train the model
history = model.fit(
    x_train.values,
    y_train['price'].values,
    epochs=10,
    batch_size=300,
    validation_split=0.2,
)

y_pred = model.predict(x_test)
# calculating mse
mse = mean_squared_error(y_test['price'], y_pred)
print('------------------------')
print('NN MSE=', round(mse, 2))
print('NN R2=', r2_score(y_test['price'], y_pred))
