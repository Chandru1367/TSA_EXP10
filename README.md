# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 19.05.2026

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('silver_prices_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
plt.plot(data['Date'], data['Price'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Silver Prices Time Series')
plt.show()
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
check_stationarity(data['Price'])
plot_acf(data['Price'])
plt.show()
plot_pacf(data['Price'])
plt.show()
sarima_model = SARIMAX(data['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
train_size = int(len(data) * 0.8)
train, test = data['Price'][:train_size], data['Price'][train_size:]
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1 )
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Model Predictions')
plt.legend()

plt.show()

```

### OUTPUT:

![alt text](image1.png)

![alt text](image2.png)

![alt text](image3.png)


![alt text](image4.png)

![alt text](image5.png)

![alt text](image6.png)

### RESULT:
Thus the program run successfully based on the SARIMA model.
