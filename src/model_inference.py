from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

def data_preprocess():

    season_dict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    month_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
             8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    
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

    X = df.drop(['cnt'], axis = 1).values
    y = df['cnt'].values

    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3, random_state = 101)

    return X_train, X_cv, y_train, y_cv

if __name__ == "__main__":

    X_train, X_cv, y_train, y_cv = data_preprocess()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    model = keras.models.load_model('models/neural_networks.h5')
    y_predictions = model.predict(X_cv)

    mse = mean_squared_error(y_cv, y_predictions)
    mae = mean_absolute_error(y_cv, y_predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

