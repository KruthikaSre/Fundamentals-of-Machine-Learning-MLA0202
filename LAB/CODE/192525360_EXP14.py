import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
data = {
'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2014, 2013, 2021, 2022, 2017, 2019, 2016, 2020, 2023],
'Kilometers': [50000, 45000, 40000, 35000, 30000, 25000, 60000, 70000, 20000, 15000, 42000, 28000, 48000, 22000, 10000],
'Price': [400000, 450000, 500000, 550000, 600000, 650000, 350000, 300000, 750000, 800000, 480000, 620000, 430000, 700000, 900000]
}
X = np.array(list(zip(data['Year'], data['Kilometers'])))
y = np.array(data['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
new_car = np.array([[2022, 18000]])
predicted_price = model.predict(new_car)[0]
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(y_test, y_pred, color='blue', label='Predicted Prices')
ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Actual Prices')
ax[0].set_xlabel("Actual Price")
ax[0].set_ylabel("Predicted Price")
ax[0].set_title("Car Price Prediction")
ax[0].legend()
ax[1].axis('off')
output = "CAR PRICE PREDICTION MODEL\n\n"
output += "Algorithm: Linear Regression\n\n"
output += "R2 Score: " + str(round(accuracy, 2)) + "\n"
output += "Mean Absolute Error: Rs. " + str(round(mae, 2)) + "\n\n"
output += "New Car Details:\n"
output += "Year: 2022\nKilometers: 18000\n\n"
output += "Predicted Car Price:\nRs. " + str(round(predicted_price, 2))
ax[1].text(0, 1, output, fontsize=11, family='monospace', verticalalignment='top')
ax[1].set_title("Console Output")
plt.tight_layout()
plt.show()
