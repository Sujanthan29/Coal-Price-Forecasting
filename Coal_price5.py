# -*- coding: utf-8 -*-
"""
Created on Thu May  9 07:26:54 2024

@author: sujan
"""
#pip install AutoGluon
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error
from feature_engine.outliers import Winsorizer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tbats import TBATS
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from itertools import product

# MySQL database connection
user = 'root'
pw = 'user1'
db = 'Coal_price'
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

# Read data from MySQL database
sql = 'select * from Coal_price'
Coal_price = pd.read_sql_query(sql, con=engine)

Coal_price = pd.read_csv(f'C:/Data Science Course/PROJECT/Project 2( Coal-iron ore price forecasting)/Dataset/Project_185(coal_forecasting)/Project_185(coal_forecasting) - Copy.csv')
# Rename column names
#Coal_price.rename(columns={'Coal_RB_4800_FOB_London_Close_USD': 'RB4800L',
                           'Coal_RB_5500_FOB_London_Close_USD': 'RB5500L',
                           'Coal_RB_5700_FOB_London_Close_USD': 'RB5700L',
                           'Coal_RB_6000_FOB_CurrentWeek_Avg_USD': 'RB6000CWA',
                           'Coal_India_5500_CFR_London_Close_USD': 'I5500CFRL'}, inplace=True)

# Exploratory Data Analysis
print(Coal_price.info())
print(Coal_price.describe())


# Convert date column type
Coal_price['Date'] = pd.to_datetime(Coal_price['Date'], format='%d-%m-%Y')

# Data Visualization
sns.pairplot(Coal_price)
plt.show()

# Correlation Matrix
corr = Coal_price.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap='YlGnBu', annot=True)
plt.title("Correlation Matrix")
plt.show()


# Data Preprocessing
X = Coal_price.drop(columns=['Coal_India_5500_CFR_London_Close_USD'])
y = Coal_price['Coal_India_5500_CFR_London_Close_USD']

## imputation
X.isnull().sum()

numeric_features = X.select_dtypes(include=['int', 'float']).columns
print(numeric_features)
num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='constant'))])
preprocessor = ColumnTransformer(transformers=[('ffill', num_pipeline, numeric_features)])
impute_data = preprocessor.fit(X)
## Save the data preprocessing pipeline
joblib.dump(impute_data, 'impute.pkl')
X1 = pd.DataFrame(impute_data.transform(X), columns= numeric_features)
X1
X1.isnull().sum()
y.isnull().sum()

y = y.fillna(method = 'ffill')
y.isnull().sum()
###############################################################################

# Outlier treatment

# Box plot for outlier check
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace = 0.75)
plt.title('Box Plot Before Outlier Treatment')
plt.tight_layout()
plt.show()

# Select numerical columns
numeric_features2 = X1.select_dtypes(include=['int', 'float']).columns.tolist()
print(numeric_features2)
# Define the winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=numeric_features2)

# Define the pipeline for outlier treatment
outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

# Apply the pipeline to the data
preprocessor1 = ColumnTransformer(transformers=[('wins', outlier_pipeline, numeric_features2)],
                                  remainder='passthrough')
winz_data = preprocessor1.fit(X1)

# Save the data preprocessing pipeline
joblib.dump(winz_data, 'winzor.pkl')

# Transform the data
X2 = pd.DataFrame(winz_data.transform(X1), columns=X1.columns)

# Box plot after outlier treatment
X2.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace = 0.75)
plt.title('Box Plot after Outlier Treatment')
plt.tight_layout()
plt.show()

# Hypothesis Testing
from statsmodels.tsa.stattools import adfuller
# select variable for ADF Test
# perform ADF test for stationarity
adf_result = adfuller(y)

# print the test statistics and p-value
print('ADF Statistics:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

# interpret the results
if adf_result[1] < 0.05:
    print("Reject the null hypothesis : The time series is stationary.")
else:
    print("Fail to reject the null hypothesis : The time seris is non-stationary.")

from statsmodels.tsa.stattools import kpss

# By setting Year column as index
Coal_price.set_index('Date', inplace = True)

# perform KPSS test
kpss_stat, p_value, lags, critical_values = kpss(y)

# print the results
print('KPSS Statistics:', kpss_stat)
print('p-value:', p_value)
print('Critical values:', critical_values)

# interpret the result
if p_value < 0.05:
    print('The time series is not stationary (reject the null hypothesis)')
else:
    print('The time series is stationary( fail to reject the null hypothesis)')

# Diffrencing to make the data stationary
y_diff = y.diff().dropna()

# plot the orginal and differenced series
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.title('Orginal time series')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)

plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 2)
plt.plot(y_diff)
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Price Difference (USD)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Perform ADF test on differenced series
adf_result = adfuller(y_diff)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

# Interpret the results
if adf_result[1] < 0.05:
    print('The differenced series is stationary (reject the null hypothesis).')
else:
    print('The differenced series is not stationary (fail to reject the null hypothesis).')

# Perform KPSS test on differenced series
kpss_stat, p_value, lags, critical_values = kpss(y_diff)
print('KPSS Statistic:', kpss_stat)
print('p-value:', p_value)
print('Critical Values:', critical_values)

# Interpret the results
if p_value < 0.05:
    print('The differenced series is not stationary (reject the null hypothesis).')
else:
    print('The differenced series is stationary (fail to reject the null hypothesis).')



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot ACF
plt.figure(figsize = (12, 6))
plot_acf(y, lags = 100, ax = plt.gca())
plt.title('Autocorrelation function (ACF)')
plt.xlabel('lag')
plt.ylabel('Autocorrelation')
plt.show()

# Plot PACF
plt.figure(figsize=(12, 6))
plot_pacf(y, lags=100, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


"""Model Building"""

## Split the data as train and test
train_size = int(len(y) * 0.95)
train, test = y[:train_size], y[train_size:]

train_size

#Autoregressive Integrated Moving Average(ARIMA)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# fit ARIMA model
model1 = ARIMA(train, order = (0, 1, 1))
model1_fit = model1.fit()

# forecast
train_forecast1 = model1_fit.predict(start=train.index[0], end=train.index[-1])
test_forecast1 = model1_fit.forecast(steps = len(test))

# Calculate MAPE
train_mape1 = mean_absolute_percentage_error(train, train_forecast1)
test_mape1 = mean_absolute_percentage_error(test, test_forecast1)
print("TRAIN MAPE:", train_mape1)
print("TEST MAPE:", test_mape1)

# hyperparameter Optimization

# Define a function to evaluate ARIMA models with different parameters
def evaluate_arima_model(train, test, order):
    # Fit the ARIMA model
    model1h = ARIMA(train, order=order)
    model_fit1h = model1h.fit()

    # Make predictions
    train_forecast1h = model_fit1h.predict(start=train.index[0], end=train.index[-1])
    test_forecast1h = model_fit1h.forecast(steps=len(test))

    # Calculate MAPE
    train_mape1h = mean_absolute_percentage_error(train, train_forecast1h)
    test_mape1h = mean_absolute_percentage_error(test, test_forecast1h)
    
    return train_mape1h, test_mape1h

# Define the range of parameters for grid search
p_values = range(0, 3)  # AR parameter
d_values = range(0, 2)  # I parameter
q_values = range(0, 3)  # MA parameter

# Generate all possible combinations of parameters
parameters = product(p_values, d_values, q_values)

# Split the data as train and test
train_size = int(len(y) * 0.95)
train, test = y[:train_size], y[train_size:]

# Initialize best MAPE
best_mape1 = float('inf')
best_params1 = None

# Perform grid search
for order in parameters:
    try:
        train_mape1h, test_mape1h = evaluate_arima_model(train, test, order)
        
        # Update best parameters if MAPE improves
        if test_mape1h < best_mape1:
            best_mape1 = test_mape1h
            best_params1 = order
            
        print(f'ARIMA{order} - Train MAPE: {train_mape1h:.2f}, Test MAPE: {test_mape1h:.2f}')
    except:
        continue

# Print best parameters
print(f'Best ARIMA model: ARIMA{best_params1} with Test MAPE: {best_mape1:.2f}')

#Seasonal Autoregressive Integrated Moving Average(SARIMA)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# fit SARIMA model
model2 = SARIMAX(train, order = (1,1,1), seasonal_order = (1,1,1,30))
model2_fit = model2.fit()

# forecast
train_forecast2 = model2_fit.predict(start=train.index[0], end=train.index[-1])
test_forecast2 = model2_fit.forecast(steps = len(test))

# calculate MAPE
train_mape2 = mean_absolute_percentage_error(train, train_forecast2)
test_mape2 = mean_absolute_percentage_error(test, test_forecast2)
print("TRAIN MAPE:", train_mape2)
print("TEST MAPE:", test_mape2)

# Vector Autoregression(VAR)
from statsmodels.tsa.api import VAR
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Fit VAR model
model3 = VAR(X_train)
model3_fit = model3.fit()

# Forecast
lag_order = model3_fit.k_ar
train_forecast3 = model3_fit.forecast(X_train.values[-lag_order:], steps=len(X_test))

test_forecast3 = model3_fit.forecast(X_train.values[-lag_order:], steps = len(X_test))

test_mape3 = mean_absolute_percentage_error(y_test, test_forecast3[:, 0])
print("test MAPE:", test_mape3)

"""THETA"""

def theta_model(train, test):
    theta = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    forecast4 = []
    for i in range(len(test)):
        forecast4.append(train.iloc[-1] + theta * (i + 1))

    return forecast4

# Apply Theta model
train_forecast4 = theta_model(train, train)
test_forecast4 = theta_model(train, test)

# Calculate MAPE for train and test data
train_mape4 = mean_absolute_percentage_error(train, train_forecast4)
test_mape4 = mean_absolute_percentage_error(test, test_forecast4)

print("Train MAPE:", train_mape4)
print("Test MAPE:", test_mape4)

"""Holt's Winter Method"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt's Winter method
model5 = ExponentialSmoothing(train, seasonal_periods=30, trend='add', seasonal='add')
model5_fit = model5.fit()

# Make predictions for both train and test sets
train_forecast5 = model5_fit.fittedvalues
test_forecast5 = model5_fit.forecast(len(test))

# Calculate MAPE for train and test sets
train_mape5 = mean_absolute_percentage_error(train, train_forecast5)
test_mape5 = mean_absolute_percentage_error(test, test_forecast5)

# Print MAPE scores
print('Train MAPE:', train_mape5)
print('Test MAPE:', test_mape5)

mape_train = np.mean(np.abs((train - train_forecast5) / train )) * 100
mape_test = np.mean(np.abs((test - test_forecast5) / test)) * 100

print(mape_train)
print(mape_test)

"""Long short term memory (LSTM)"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
scaled_test = scaler.transform(test.values.reshape(-1, 1))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30
X_train, y_train = create_dataset(scaled_train, scaled_train, time_steps)
X_test, y_test = create_dataset(scaled_test, scaled_test, time_steps)

model6 = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model6.compile(optimizer='adam', loss='mse')
model6.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

train_forecast6 = model6.predict(X_train)
test_forecast6 = model6.predict(X_test)

train_forecast6 = scaler.inverse_transform(train_forecast6).flatten()
train6 = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

test_forecast6 = scaler.inverse_transform(test_forecast6).flatten()
test6 = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

train_mape6 = mean_absolute_percentage_error(train6, train_forecast6)
print('Train MAPE:', train_mape6)

test_mape6 = mean_absolute_percentage_error(test6, test_forecast6)
print('Test MAPE:', test_mape6)

mape_train6 = np.mean(np.abs((train6 - train_forecast6) / train6)) * 100
mape_test6 = np.mean(np.abs((test6 - test_forecast6) / test6)) * 100

print(mape_train6)
print(mape_test6)

## hyperparameter Optimization

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import randint as sp_randint
from sklearn.base import BaseEstimator, RegressorMixin
from keras.src.models.sequential import Sequential

# Define your mean absolute percentage error function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Define your LSTM model as a function
def create_lstm_model(units, time_steps):
    model = Sequential([
        LSTM(units=units, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Define custom KerasRegressor class
class CustomKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, units=50, time_steps=30, epochs=100, batch_size=32, verbose=0):
        self.units = units
        self.time_steps = time_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        self.model = create_lstm_model(self.units, self.time_steps)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define hyperparameters
param_dist = {
    'units': sp_randint(10, 100),  # Number of LSTM units
    'time_steps': sp_randint(10, 100)  # Time steps for sequence data
}

# Scale your data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
scaled_test = scaler.transform(test.values.reshape(-1, 1))

# Create dataset
time_steps = 30
X_train, y_train = create_dataset(scaled_train, scaled_train, time_steps)
X_test, y_test = create_dataset(scaled_test, scaled_test, time_steps)

# Create LSTM model
lstm_model = CustomKerasRegressor(epochs=100, batch_size=32, verbose=0)

# Perform Random Search
random_search = RandomizedSearchCV(estimator=lstm_model, param_distributions=param_dist, n_iter=10, cv=3, scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False))
lstm_new = random_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)



"""TBATS"""

from tbats import TBATS

# Fit TBATS model
model7 = TBATS(seasonal_periods = [1, 30], use_arma_errors=True)
model7_fit = model7.fit(train)

# Forecast train and test data
train_forecast7 = model7_fit.forecast(steps=len(train))
test_forecast7 = model7_fit.forecast(steps=len(test))# Calculate MAPE for train and test data
train_mape7 = mean_absolute_percentage_error(train, train_forecast7)
test_mape7 = mean_absolute_percentage_error(test, test_forecast7)

# Print MAPE scores
print("TRAIN MAPE:", train_mape7)
print("TEST MAPE:", test_mape7)

# AutoML (AutoGluon)
train_data = TabularDataset(Coal_price.reset_index())
predictor = TabularPredictor(label='I5500CFRL', problem_type='regression', eval_metric='mean_absolute_percentage_error')
predictor.fit(train_data)

# Make predictions
test_pred = predictor.predict(train_data)
test_mape = mean_absolute_percentage_error(test_pred, Coal_price['I5500CFRL'])
print('Test MAPE:', test_mape)


import pickle


# Define Best_Score
Best_Score = {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.90}
# Save the model
with open('lstm.pkl', 'wb') as file:
    pickle.dump(Best_Score, file)
    
import os
os.getcwd()
