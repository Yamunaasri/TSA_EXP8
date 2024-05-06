# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
```
data = pd.read_csv("/content/airline.csv")
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))
plt.plot(data['International '].head(50))
plt.title('First 50 values of the "International" column')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.show()
```
```
rolling_mean_5 = data['International '].rolling(window=5).mean()
print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))
rolling_mean_10 = data['International '].rolling(window=10).mean()
plt.plot(data['International '], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```
```
lag_order = 13
model = AutoReg(data['International '], lags=lag_order)
model_fit = model.fit()
plot_acf(data['International '])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['International '])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
predictions = model_fit.predict(start=lag_order, end=len(data)-1)
mse = mean_squared_error(data['International '][lag_order:], predictions)
print('Mean Squared Error (MSE):', mse)
plt.plot(data['International '][lag_order:], label='Original Data')
plt.plot(predictions, label='Predictions')
plt.title('AR Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```
### OUTPUT:

#### Plot the original data and fitted value
![image](https://github.com/Yamunaasri/TSA_EXP8/assets/115707860/29d67fc1-b8fc-4ae8-8309-270e40b0d51e)
#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
![image](https://github.com/Yamunaasri/TSA_EXP8/assets/115707860/e08dc041-fb20-4b92-a018-2788f6db34af)

![image](https://github.com/Yamunaasri/TSA_EXP8/assets/115707860/31f0f98e-6d92-4b3a-8c2c-454070acc95f)

#### Plot the original data and predictions
![image](https://github.com/Yamunaasri/TSA_EXP8/assets/115707860/3c0c9eda-65ef-4c89-b84a-f992df4bbd5a)

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
