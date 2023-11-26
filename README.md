# AI Project

import pandas as pd

temperature = pd.read_csv('/content/Sheet 1-GlobalLandTemperaturesByMajorCity.csv')
temperature

temperature_Myanmar = temperature[temperature['Country'] == 'Burma']
temperature_Myanmar

temperature_Myanmar.shape

temperature_Myanmar.isnull()

temperature_Myanmar.isnull().sum()

null_value_percentage = (temperature_Myanmar.isnull().sum() * 100) / len(temperature)
null_value_percentage

temperature_null_drop = temperature_Myanmar.dropna(how = 'any')
temperature_null_drop.isnull().sum()

temperature1 = temperature_null_drop
temperature1

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(temperature1['dt'], temperature1['AverageTemperature'])

plt.scatter(temperature1['dt'], temperature1['AverageTemperatureUncertainty'])

x = temperature1[['dt', 'AverageTemperatureUncertainty']]
y = temperature1['AverageTemperature']
x
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

len(x_train)
len(x_test)

x_train
y_train

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
y_predict

mae = sum([abs(i-j) for i,j in zip(y_test,y_predict)]) / len(y_test)
print("Mean Absolute Error =", made)

