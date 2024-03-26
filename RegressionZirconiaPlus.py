import numpy as np
import seaborn as sns
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
#### 3. Encode Ordinal features
#cuts = [['Fair'], ['Good'], ['Very Good'], ['Premium'], ['Ideal'], ['Fair']]
## 3.a Ordinal Encoding
ordinal = data[['cut', 'color', 'clarity']]
# # Worst -> Best: 0, 1, 2...
encoder = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                     ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                     ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']])
# # Fit and transform the data
data[['cut', 'color', 'clarity']] = encoder.fit_transform(ordinal)
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
x_train, x_test, y_train, y_test = train_test_split (data.iloc[:,:],data.iloc[:,-2:], test_size=0.3, random_state=42)
# Original Heatmap
plt.figure(10)
corr_matrix = data.corr()
f, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr_matrix,
            annot=True,
            annot_kws={'size': 8},
            cmap="Spectral_r")

#### 4.5 Find splits for
##print(x_train[['cut', 'price_carat']].groupby('cut').mean())
#      price_carat
# cut
# 0.0  4178.218294
# 1.0  4075.386070
# 2.0  4089.093958
# 3.0  4371.447664
# 4.0  3836.810796
x_train.loc[x_train['cut']==0.0, 'cut'] = 4178.218294
x_train.loc[x_train['cut']==1.0, 'cut'] = 4075.386070
x_train.loc[x_train['cut']==2.0, 'cut'] = 4089.093958
x_train.loc[x_train['cut']==3.0, 'cut'] = 4371.447664
x_train.loc[x_train['cut']==4.0, 'cut'] = 3836.810796
x_test.loc[x_test['cut']==0.0, 'cut'] = 4178.218294
x_test.loc[x_test['cut']==1.0, 'cut'] = 4075.386070
x_test.loc[x_test['cut']==2.0, 'cut'] = 4089.093958
x_test.loc[x_test['cut']==3.0, 'cut'] = 4371.447664
x_test.loc[x_test['cut']==4.0, 'cut'] = 3836.810796
#print(x_train[['cut', 'price']].groupby('cut').mean())
##print(x_train[['color', 'price_carat']].groupby('color').mean())
#        price_carat
# color
# 0.0    4074.421647
# 1.0    4242.263410
# 2.0    4198.260033
# 3.0    4243.583535
# 4.0    4064.675889
# 5.0    3722.898515
# 6.0    3757.711006
x_train.loc[x_train['color']==0.0, 'color'] = 4074.421647
x_train.loc[x_train['color']==1.0, 'color'] = 4242.263410
x_train.loc[x_train['color']==2.0, 'color'] = 4198.260033
x_train.loc[x_train['color']==3.0, 'color'] = 4243.583535
x_train.loc[x_train['color']==4.0, 'color'] = 4064.675889
x_train.loc[x_train['color']==3.0, 'color'] = 3722.898515
x_train.loc[x_train['color']==4.0, 'color'] = 3757.711006
x_test.loc[x_test['color']==0.0, 'color'] = 4074.421647
x_test.loc[x_test['color']==1.0, 'color'] = 4242.263410
x_test.loc[x_test['color']==2.0, 'color'] = 4198.260033
x_test.loc[x_test['color']==3.0, 'color'] = 4243.583535
x_test.loc[x_test['color']==4.0, 'color'] = 4064.675889
x_test.loc[x_test['color']==3.0, 'color'] = 3722.898515
x_test.loc[x_test['color']==4.0, 'color'] = 3757.711006
#print(x_train[['color', 'price']].groupby('color').mean())
##print(x_train[['clarity', 'price_carat']].groupby('clarity').mean())
#          price_carat
# clarity
# 2.0      3052.792088
# 3.0      4186.055291
# 4.0      3987.874720
# 5.0      4158.849531
# 6.0      4130.439523
# 7.0      3946.810601
# 8.0      3442.975768
# 9.0      3673.568498
x_train.loc[x_train['clarity']==2.0, 'clarity'] = 3052.792088
x_train.loc[x_train['clarity']==3.0, 'clarity'] = 4186.055291
x_train.loc[x_train['clarity']==4.0, 'clarity'] = 3987.874720
x_train.loc[x_train['clarity']==5.0, 'clarity'] = 4158.849531
x_train.loc[x_train['clarity']==6.0, 'clarity'] = 4130.439523
x_train.loc[x_train['clarity']==7.0, 'clarity'] = 3946.810601
x_train.loc[x_train['clarity']==8.0, 'clarity'] = 3442.975768
x_train.loc[x_train['clarity']==9.0, 'clarity'] = 3673.568498
x_test.loc[x_test['clarity']==2.0, 'clarity'] = 3052.792088
x_test.loc[x_test['clarity']==3.0, 'clarity'] = 4186.055291
x_test.loc[x_test['clarity']==4.0, 'clarity'] = 3987.874720
x_test.loc[x_test['clarity']==5.0, 'clarity'] = 4158.849531
x_test.loc[x_test['clarity']==6.0, 'clarity'] = 4130.439523
x_test.loc[x_test['clarity']==7.0, 'clarity'] = 3946.810601
x_test.loc[x_test['clarity']==8.0, 'clarity'] = 3442.975768
x_test.loc[x_test['clarity']==9.0, 'clarity'] = 3673.568498
# Update these values to the original data
#print(x_train.head())
#print(x_test.head())
# New Heatmap
plt.figure(20)
corr_matrix = x_train.corr()
f, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(corr_matrix,
            annot=True,
            annot_kws={'size': 8},
            cmap="Spectral_r")
###plt.show()
### Remove 'price', 'price_carat' columns
x_train.drop(columns=['price', 'price_carat'], inplace=True)
x_test.drop(columns=['price', 'price_carat'], inplace=True)

#print(data[['clarity', 'price']].groupby('clarity').mean())

# cut_average = []
# for cut in ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']:
#     cut_average.append()
# tmp = np.avex_train[x_train['cut'] == 'Fair']['price_carat']
# np.average(tmp)
#exit(0)
# #x_train, x_test, y_train, y_test = train_test_split (data.iloc[:,:-2],data.iloc[:,-2:], test_size=0.3, random_state=23)
# # Verify that the split gives similar distributions between train and test on the Ordinal columns
# try:
#     print(x_train['cut'].value_counts(normalize=True), x_test['cut'].value_counts(normalize=True))
#     print(x_train['color'].value_counts(normalize=True), x_test['color'].value_counts(normalize=True))
#     print(x_train['clarity'].value_counts(normalize=True), x_test['clarity'].value_counts(normalize=True))
# except:
#     pass
# # Also show in plots
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
# #### 5. Investigate the correlation of the data
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
# Let's see the correlation again
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
# #### 6. Do a simple linear regression for each column
# for col in x_train.columns:
#     # Create the model
#     linear_regression = LinearRegression()
#     # training the linear model
#     linear_regression.fit(x_train[[col]], y_train['price'])
#     # Predict
#     y_pred = linear_regression.predict(x_test[[col]])
#     # calculating mse
#     mse = mean_squared_error(y_test['price'], y_pred)
#     print('------------------------')
#     print(col, 'linear MSE=', round(mse, 2))
#     print(col, 'linear R2=', r2_score(y_test['price'], y_pred))
# # For all columns
# # Create the model
# linear_regression = LinearRegression()
# # training the linear model
# linear_regression.fit(x_train, y_train['price'])
# # Predict
# y_pred = linear_regression.predict(x_test)
# # calculating mse
# mse = mean_squared_error(y_test['price'], y_pred)
# print('------------------------')
# print('All columns', 'linear MSE=', round(mse, 2))
# print('All columns', 'linear R2=', r2_score(y_test['price'], y_pred))
# # Plot outputs
# plt.figure(1)
# plt.title('Linear Regression')
# plt.ylabel('yyy')
# plt.xlabel('Input feature=carat')
# plt.scatter(x_test[:100]['carat'], y_test[:100]['price'],  color='black')
# plt.scatter(x_test[:100]['carat'], y_pred[:100], color='blue')
# plt.xticks(())
# plt.yticks(())
# plt.show()
# # Try PCA (NOPE!)
# pca = PCA()
# fit_data = pd.DataFrame(pca.fit_transform(data.iloc[:,1:-1]))
# print(pca.explained_variance_ratio_)
# x_train, x_test, y_train, y_test = train_test_split (fit_data,data.iloc[:,-1:], test_size=0.3, random_state=42)
# # Create the model
# linear_regression = LinearRegression()
# # training the linear model
# linear_regression.fit(x_train, y_train)
# # Predict
# y_pred = linear_regression.predict(x_test)
# # calculating mse
# mse = np.mean((y_pred - y_test) ** 2)
# mse = mean_squared_error(y_test, y_pred)
# print('------------------------')
# print('All columns PCA', 'linear MSE=', round(mse, 2))
# print('All columns PCA', 'linear R2=', r2_score(y_test, y_pred))
# #### 7. Do Polynomial linear regression for each column
# for col in x_train.columns:
#     # Create the model
#     linear_regression = LinearRegression()
#
#     poly = PolynomialFeatures(degree=2)
#     # features for poly regression
#     #x_poly = poly.fit_transform(x_train[[col]])
#     # training the linear model
#     linear_regression.fit(poly.fit_transform(x_train[[col]]), y_train)
#     # Predict
#     y_pred = linear_regression.predict(poly.fit_transform(x_test[[col]]))
#     # calculating mse
#     mse = np.mean((y_pred - y_test) ** 2)
#     mse = mean_squared_error(y_test, y_pred)
#     print('------------------------')
#     print(col, 'polynomial MSE=', round(mse, 2))
#     print(col, 'polynomial R2=', r2_score(y_test, y_pred))
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
# #########################
# #### 6. Do a simple linear regression for each column but for price/carat
# for col in x_train.columns:
#     # Create the model
#     linear_regression = LinearRegression()
#     # training the linear model
#     linear_regression.fit(x_train[[col]], y_train['price_carat'])
#     # Predict
#     y_pred = linear_regression.predict(x_test[[col]])
#     # calculating mse
#     mse = mean_squared_error(y_test['price_carat'], y_pred)
#     print('------------------------')
#     print(col, ' price/carat linear MSE=', round(mse, 2))
#     print(col, ' price/carat linear R2=', r2_score(y_test['price_carat'], y_pred))
# # For all columns
# # Create the model
# linear_regression = LinearRegression()
# # training the linear model
# linear_regression.fit(x_train.drop(columns=['carat']), y_train['price_carat'])
# # Predict
# y_pred = linear_regression.predict(x_test.drop(columns=['carat']))
# # calculating mse
# mse = mean_squared_error(y_test['price_carat'], y_pred)
# print('------------------------')
# print('All columns price/carat', 'linear MSE=', round(mse, 2))
# print('All columns price/carat', 'linear R2=', r2_score(y_test['price_carat'], y_pred))
# #### 7. Do Polynomial linear regression for each column but for price/carat
# for col in x_train.columns:
#     # Create the model
#     linear_regression = LinearRegression()
#
#     poly = PolynomialFeatures(degree=2)
#     # features for poly regression
#     #x_poly = poly.fit_transform(x_train[[col]])
#     # training the linear model
#     linear_regression.fit(poly.fit_transform(x_train[[col]]), y_train['price_carat'])
#     # Predict
#     y_pred = linear_regression.predict(poly.fit_transform(x_test[[col]]))
#     # calculating mse
#     mse = np.mean((y_pred - y_test['price_carat']) ** 2)
#     mse = mean_squared_error(y_test['price_carat'], y_pred)
#     print('------------------------')
#     print(col, 'polynomial price_carat MSE=', round(mse, 2))
#     print(col, 'polynomial price_carat R2=', r2_score(y_test['price_carat'], y_pred))
# print('# Polynomial regression: ###################')
# for degree in degrees:
#     # For all columns
#     # Create the model
#     linear_regression = LinearRegression()
#     poly = PolynomialFeatures(degree=degree)
#     # training the linear model
#     linear_regression.fit(poly.fit_transform(x_train.drop(columns=['carat'])), y_train['price_carat'])
#     # Predict
#     y_pred = linear_regression.predict(poly.fit_transform(x_test.drop(columns=['carat'])))
#     # calculating mse
#     mse = mean_squared_error(y_test['price'], y_pred)
#     print('------------------------')
#     print('All columns price/carat', 'polynomial', degree, 'MSE=', round(mse, 2))
#     print('All columns price/carat', 'polynomial', degree, 'R2=', r2_score(y_test['price_carat'], y_pred))
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
#poly = PolynomialFeatures(degree=2)
ridge_regression = Ridge(alpha=0.1, max_iter=100000)
ridge_regression.fit(poly.fit_transform(x_train), y_train['price'])
y_pred = ridge_regression.predict(poly.fit_transform(x_test))
# calculating mse
mse = mean_squared_error(y_test['price'], y_pred)
print('------------------------')
print('All columns Ridge MSE=', round(mse, 2))
print('All columns Ridge R2=', r2_score(y_test['price'], y_pred))
# #### 9 Lasso
# poly = PolynomialFeatures(degree=5)
# #poly = PolynomialFeatures(degree=2)
# lasso_regression = Lasso(alpha=0.1)
# lasso_regression.fit(poly.fit_transform(x_train), y_train['price'])
# y_pred = lasso_regression.predict(poly.fit_transform(x_test))
# # calculating mse
# mse = mean_squared_error(y_test['price'], y_pred)
# print('------------------------')
# print('All columns Lasso MSE=', round(mse, 2))
# print('All columns Lasso R2=', r2_score(y_test['price'], y_pred))
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

# #### 11 NN regression
# from tensorflow.keras import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanSquaredError
#
#
# learning_rate = 0.01
# # Creating model using the Sequential in tensorflow
# model = Sequential([
#     Dense(100, kernel_initializer='normal', activation='relu'),
#     Dropout(0.2),
#     Dense(200, kernel_initializer='normal', activation='relu'),
#     Dropout(0.2),
#     Dense(50, kernel_initializer='normal', activation='relu'),
#     Dense(1, kernel_initializer='normal', activation='linear')
# ])
#
# # loss function
# msle = MeanSquaredLogarithmicError()
# mse = MeanSquaredError()
# model.compile(
#     loss=msle,
#     optimizer=Adam(learning_rate=learning_rate),
#     #metrics=[msle],
#     metrics = [mse],
# )
#
# # train the model
# history = model.fit(
#     x_train.values,
#     y_train['price'].values,
#     epochs=10,
#     batch_size=300,
#     validation_split=0.2,
# )
#
# y_pred = model.predict(x_test)
# # calculating mse
# mse = mean_squared_error(y_test['price'], y_pred)
# print('------------------------')
# print('NN MSE=', round(mse, 2))
# print('NN R2=', r2_score(y_test['price'], y_pred))

exit(0)
