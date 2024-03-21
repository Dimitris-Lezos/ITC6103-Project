import numpy as np
import sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd


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
# # Very poor results
# # Instantiate the encoder
# encoder = ce.HelmertEncoder(cols=['cut', 'color', 'clarity'])
# # Fit and transform the data
# df_encoded = encoder.fit_transform(data[['cut', 'color', 'clarity']])
# price = data[['price']]
# data.drop(columns=['price'], inplace=True)
# data = data.join(df_encoded)
# data = data.join(price)
# data.drop(columns=['cut', 'intercept', 'id', 'color', 'clarity'], inplace=True)
# # Print the encoded data
# print(data.head())
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
data['price_carat'] = data['price'] / data['carat']
x_train, x_test, y_train, y_test = train_test_split (data.iloc[:,:-2],data.iloc[:,-2:], test_size=0.3, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split (data.iloc[:,:-2],data.iloc[:,-2:], test_size=0.3, random_state=23)
# Verify that the split gives similar distributions between train and test on the Ordinal columns
try:
    print(x_train['cut'].value_counts(normalize=True), x_test['cut'].value_counts(normalize=True))
    print(x_train['color'].value_counts(normalize=True), x_test['color'].value_counts(normalize=True))
    print(x_train['clarity'].value_counts(normalize=True), x_test['clarity'].value_counts(normalize=True))
except:
    pass
# Also show in plots
plt.figure(1)
plt.plot(x_train['cut'].value_counts(normalize=True, sort=False).sort_index()+0.02)
plt.plot(x_test['cut'].value_counts(normalize=True, sort=False).sort_index())
plt.figure(2)
plt.plot(x_train['color'].value_counts(normalize=True, sort=False).sort_index()+0.02)
plt.plot(x_test['color'].value_counts(normalize=True, sort=False).sort_index())
plt.figure(3)
plt.plot(x_train['clarity'].value_counts(normalize=True, sort=False).sort_index()+0.02)
plt.plot(x_test['clarity'].value_counts(normalize=True, sort=False).sort_index())
plt.show()
#### 5. Investigate the correlation of the data
print(data.describe())
corr_matrix = data.corr()
f, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr_matrix,
            annot=True,
            annot_kws={'size': 8},
            cmap="Spectral_r")
plt.show()
# We see that carat, x, y, z are completely correlated, so we remove x, y & z features
data.drop(columns=['x', 'y', 'z'], inplace=True)
x_train.drop(columns=['x', 'y', 'z'], inplace=True)
x_test.drop(columns=['x', 'y', 'z'], inplace=True)
# Let's see the correlation again
corr_matrix = data.corr()
f, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr_matrix,
            annot=True,
            annot_kws={'size': 8},
            cmap="Spectral_r")
plt.show()
# Price and carat are very closely correlated as expected we will try to train for two cases:
# a. Price
# b. Price / carat
#### 5.5 Boxplots
plt.figure(4)
plot = sns.boxplot(data=data.drop(columns=['id', 'price', 'price_carat']))
plt.show()
#### 6. Do a simple linear regression for each column
for col in x_train.columns:
    # Create the model
    linear_regression = LinearRegression()
    # training the linear model
    linear_regression.fit(x_train[[col]], y_train['price'])
    # Predict
    y_pred = linear_regression.predict(x_test[[col]])
    # calculating mse
    mse = mean_squared_error(y_test['price'], y_pred)
    print('------------------------')
    print(col, 'linear MSE=', round(mse, 2))
    print(col, 'linear R2=', r2_score(y_test['price'], y_pred))
# For all columns
# Create the model
linear_regression = LinearRegression()
# training the linear model
linear_regression.fit(x_train, y_train['price'])
# Predict
y_pred = linear_regression.predict(x_test)
# calculating mse
mse = mean_squared_error(y_test['price'], y_pred)
print('------------------------')
print('All columns', 'linear MSE=', round(mse, 2))
print('All columns', 'linear R2=', r2_score(y_test['price'], y_pred))
# Plot outputs
plt.figure(1)
plt.title('Linear Regression')
plt.ylabel('yyy')
plt.xlabel('Input feature=carat')
plt.scatter(x_test[:100]['carat'], y_test[:100]['price'],  color='black')
plt.scatter(x_test[:100]['carat'], y_pred[:100], color='blue')
plt.xticks(())
plt.yticks(())
plt.show()
# Try PCA (NOPE!)
pca = PCA()
fit_data = pd.DataFrame(pca.fit_transform(data.iloc[:,1:-1]))
print(pca.explained_variance_ratio_)
x_train, x_test, y_train, y_test = train_test_split (fit_data,data.iloc[:,-1:], test_size=0.3, random_state=42)
# Create the model
linear_regression = LinearRegression()
# training the linear model
linear_regression.fit(x_train, y_train)
# Predict
y_pred = linear_regression.predict(x_test)
# calculating mse
mse = np.mean((y_pred - y_test) ** 2)
mse = mean_squared_error(y_test, y_pred)
print('------------------------')
print('All columns PCA', 'linear MSE=', round(mse, 2))
print('All columns PCA', 'linear R2=', r2_score(y_test, y_pred))
#### 7. Do Polynomial linear regression for each column
for col in x_train.columns:
    # Create the model
    linear_regression = LinearRegression()

    poly = PolynomialFeatures(degree=2)
    # features for poly regression
    #x_poly = poly.fit_transform(x_train[[col]])
    # training the linear model
    linear_regression.fit(poly.fit_transform(x_train[[col]]), y_train)
    # Predict
    y_pred = linear_regression.predict(poly.fit_transform(x_test[[col]]))
    # calculating mse
    mse = np.mean((y_pred - y_test) ** 2)
    mse = mean_squared_error(y_test, y_pred)
    print('------------------------')
    print(col, 'polynomial MSE=', round(mse, 2))
    print(col, 'polynomial R2=', r2_score(y_test, y_pred))
print('# Polynomial regression: ###################')
#degrees = range(2,7)
degrees = range(5,6)
#degrees = range(2,3)
for degree in degrees:
    # For all columns
    # Create the model
    linear_regression = LinearRegression()
    poly = PolynomialFeatures(degree=degree)
    # training the linear model
    linear_regression.fit(poly.fit_transform(x_train), y_train['price'])
    # Predict
    y_pred = linear_regression.predict(poly.fit_transform(x_test))
    # calculating mse
    mse = mean_squared_error(y_test['price'], y_pred)
    print('------------------------')
    print('All columns', 'polynomial', degree, 'MSE=', round(mse, 2))
    print('All columns', 'polynomial', degree, 'R2=', r2_score(y_test['price'], y_pred))
#########################
#### 6. Do a simple linear regression for each column but for price/carat
for col in x_train.columns:
    # Create the model
    linear_regression = LinearRegression()
    # training the linear model
    linear_regression.fit(x_train[[col]], y_train['price_carat'])
    # Predict
    y_pred = linear_regression.predict(x_test[[col]])
    # calculating mse
    mse = mean_squared_error(y_test['price_carat'], y_pred)
    print('------------------------')
    print(col, ' price/carat linear MSE=', round(mse, 2))
    print(col, ' price/carat linear R2=', r2_score(y_test['price_carat'], y_pred))
# For all columns
# Create the model
linear_regression = LinearRegression()
# training the linear model
linear_regression.fit(x_train.drop(columns=['carat']), y_train['price_carat'])
# Predict
y_pred = linear_regression.predict(x_test.drop(columns=['carat']))
# calculating mse
mse = mean_squared_error(y_test['price_carat'], y_pred)
print('------------------------')
print('All columns price/carat', 'linear MSE=', round(mse, 2))
print('All columns price/carat', 'linear R2=', r2_score(y_test['price_carat'], y_pred))
#### 7. Do Polynomial linear regression for each column but for price/carat
for col in x_train.columns:
    # Create the model
    linear_regression = LinearRegression()

    poly = PolynomialFeatures(degree=2)
    # features for poly regression
    #x_poly = poly.fit_transform(x_train[[col]])
    # training the linear model
    linear_regression.fit(poly.fit_transform(x_train[[col]]), y_train['price_carat'])
    # Predict
    y_pred = linear_regression.predict(poly.fit_transform(x_test[[col]]))
    # calculating mse
    mse = np.mean((y_pred - y_test['price_carat']) ** 2)
    mse = mean_squared_error(y_test['price_carat'], y_pred)
    print('------------------------')
    print(col, 'polynomial price_carat MSE=', round(mse, 2))
    print(col, 'polynomial price_carat R2=', r2_score(y_test['price_carat'], y_pred))
print('# Polynomial regression: ###################')
for degree in degrees:
    # For all columns
    # Create the model
    linear_regression = LinearRegression()
    poly = PolynomialFeatures(degree=degree)
    # training the linear model
    linear_regression.fit(poly.fit_transform(x_train.drop(columns=['carat'])), y_train['price_carat'])
    # Predict
    y_pred = linear_regression.predict(poly.fit_transform(x_test.drop(columns=['carat'])))
    # calculating mse
    mse = mean_squared_error(y_test['price'], y_pred)
    print('------------------------')
    print('All columns price/carat', 'polynomial', degree, 'MSE=', round(mse, 2))
    print('All columns price/carat', 'polynomial', degree, 'R2=', r2_score(y_test['price_carat'], y_pred))
#########################################
#### 6. Do a simple linear regression for columns carat, clarity, table
# # for col in x_train.columns:
# #     # Create the model
# #     linear_regression = LinearRegression()
# #     # training the linear model
# #     linear_regression.fit(x_train[[col]], y_train['price'])
# #     # Predict
# #     y_pred = linear_regression.predict(x_test[[col]])
# #     # calculating mse
# #     mse = mean_squared_error(y_test['price'], y_pred)
# #     print('------------------------')
# #     print(col, 'linear MSE=', round(mse, 2))
# #     print(col, 'linear R2=', r2_score(y_test['price'], y_pred))
# # For all columns
# # Create the model
# linear_regression = LinearRegression()
# # training the linear model
# linear_regression.fit(x_train.drop(columns=['cut', 'color', 'depth']), y_train['price'])
# # Predict
# y_pred = linear_regression.predict(x_test.drop(columns=['cut', 'color', 'depth']))
# # calculating mse
# mse = mean_squared_error(y_test['price'], y_pred)
# print('------------------------')
# print('All [carat, clarity, table]', 'linear MSE=', round(mse, 2))
# print('All [carat, clarity, table]', 'linear R2=', r2_score(y_test['price'], y_pred))
# #### 7. Do Polynomial linear regression for each column
# # for col in x_train.columns:
# #     # Create the model
# #     linear_regression = LinearRegression()
# #
# #     poly = PolynomialFeatures(degree=2)
# #     # features for poly regression
# #     #x_poly = poly.fit_transform(x_train[[col]])
# #     # training the linear model
# #     linear_regression.fit(poly.fit_transform(x_train[[col]]), y_train)
# #     # Predict
# #     y_pred = linear_regression.predict(poly.fit_transform(x_test[[col]]))
# #     # calculating mse
# #     mse = np.mean((y_pred - y_test) ** 2)
# #     mse = mean_squared_error(y_test, y_pred)
# #     print('------------------------')
# #     print(col, 'polynomial MSE=', round(mse, 2))
# #     print(col, 'polynomial R2=', r2_score(y_test, y_pred))
# print('# Polynomial regression: ###################')
# for degree in degrees:
#     # For all columns
#     # Create the model
#     linear_regression = LinearRegression()
#     poly = PolynomialFeatures(degree=degree)
#     # training the linear model
#     linear_regression.fit(poly.fit_transform(x_train.drop(columns=['cut', 'color', 'depth'])), y_train['price'])
#     # Predict
#     y_pred = linear_regression.predict(poly.fit_transform(x_test.drop(columns=['cut', 'color', 'depth'])))
#     # calculating mse
#     mse = mean_squared_error(y_test['price'], y_pred)
#     print('------------------------')
#     print('All [carat, clarity, table]', 'polynomial', degree, 'MSE=', round(mse, 2))
#     print('All [carat, clarity, table]', 'polynomial', degree, 'R2=', r2_score(y_test['price'], y_pred))
#### 8 Ridge
poly = PolynomialFeatures(degree=5)
ridge_regression = Ridge(alpha=0.1)
ridge_regression.fit(poly.fit_transform(x_train), y_train['price'])
y_pred = ridge_regression.predict(poly.fit_transform(x_test))
# calculating mse
mse = mean_squared_error(y_test['price'], y_pred)
print('------------------------')
print('All columns Ridge MSE=', round(mse, 2))
print('All columns Ridge R2=', r2_score(y_test['price'], y_pred))
#### 9 Lasso
poly = PolynomialFeatures(degree=5)
lasso_regression = Lasso(alpha=0.1)
lasso_regression.fit(poly.fit_transform(x_train), y_train['price'])
y_pred = lasso_regression.predict(poly.fit_transform(x_test))
# calculating mse
mse = mean_squared_error(y_test['price'], y_pred)
print('------------------------')
print('All columns Lasso MSE=', round(mse, 2))
print('All columns Lasso R2=', r2_score(y_test['price'], y_pred))
exit(0)
#### 10 Backward Stepwise Polynomial regression
# print('# Polynomial regression: ###################')
x_train_drop = x_train.copy()
x_test_drop = x_test.copy()

for column in ['depth', 'cut', 'table', 'clarity', 'color']:
    x_train_drop.drop(columns=[column], inplace=True)
    x_test_drop.drop(columns=[column], inplace=True)
    for degree in degrees: #range(2,6):
        # For all columns
        # Create the model
        linear_regression = LinearRegression()
        poly = PolynomialFeatures(degree=degree)
        # training the linear model
        linear_regression.fit(poly.fit_transform(x_train_drop), y_train['price'])
        # Predict
        y_pred = linear_regression.predict(poly.fit_transform(x_test_drop))
        # calculating mse
        mse = mean_squared_error(y_test['price'], y_pred)
        print('------------------------')
        print(x_train_drop.columns, 'polynomial', degree, 'MSE=', round(mse, 2))
        print(x_train_drop.columns, 'polynomial', degree, 'R2=', r2_score(y_test['price'], y_pred))



exit(0)
