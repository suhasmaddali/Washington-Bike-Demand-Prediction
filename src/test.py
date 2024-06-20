import sys
import pprint
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from datetime import datetime
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")


def temperature_converter(value):

    temp_max = df['temp'].max()
    temp_min = df['temp'].min()

    return (value * (temp_max - temp_min) + temp_min)

def temperature_segments(value):
    if value < 0.2:
        return 0.1
    elif value < 0.4 and value > 0.2:
        return 0.3
    elif value < 0.6 and value > 0.4:
        return 0.5
    elif value < 0.7 and value > 0.5:
        return 0.6
    elif value < 0.8 and value > 0.6:
        return 0.7
    elif value < 0.9 and value > 0.7:
        return 0.8
    else:
        return 0.9

if __name__ == "__main__":

    season_dict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    month_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
             8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

    df = pd.read_csv('data/hour.csv')
    df['Season_cat'] = df['season'].map(season_dict)
    df["Month_cat"] = df['mnth'].map(month_dict)
    df['day']= df['dteday'].apply(lambda x: x[8:])
    df['Temperature_converted'] = df['temp'].apply(temperature_converter)
    df['Temperature_segments'] = df['temp'].apply(temperature_segments)
    df.drop(['Season_cat', 'Month_cat'], axis = 1, inplace = True)
    df['day'] = df['day'].apply(lambda x: int(x))
    df.drop(['casual', 'registered'], axis = 1, inplace = True)
    df['dteday'] = df['dteday'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df.drop(['instant', 'dteday'], axis = 1, inplace = True)

    print(df.head())

    X = df.drop(['cnt'], axis = 1).values
    y = df['cnt'].values

    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3, random_state = 101)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    model = Sequential()
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'MSE', metrics = ['MSE', 'MAE'])

    model.fit(X_train, y_train, epochs = 10, verbose = 1, validation_data = (X_cv, y_cv))

