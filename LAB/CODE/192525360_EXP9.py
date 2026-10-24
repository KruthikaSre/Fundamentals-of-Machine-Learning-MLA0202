import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 5, 7, 10, 15, 19, 25, 32])
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear = linear_model.predict(X)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly = poly_model.predict(X_poly)
linear_mse = np.mean((y - y_linear) ** 2)
poly_mse = np.mean((y - y_poly) ** 2)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X, y, label="Actual Data")
ax.plot(X, y_linear, label="Linear Regression")
ax.plot(X, y_poly, label="Polynomial Regression")
ax.set_title("Linear vs Polynomial Regression")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
output = (
    "CONSOLE OUTPUT\n\n"
    f"Linear Regression MSE: {linear_mse:.3f}\n"
    f"Polynomial Regression MSE: {poly_mse:.3f}\n\n"
    f"Linear Equation: y = {linear_model.coef_[0]:.3f}x "
    f"+ {linear_model.intercept_:.3f}\n"
    f"Polynomial Degree: 2"
)
fig.text(
    0.5, 0.02, output,
    ha="center", va="bottom",
    fontsize=10, family="monospace",
    bbox=dict(boxstyle="round", facecolor="white")
)
plt.tight_layout(rect=[0, 0.20, 1, 1])
plt.show()
