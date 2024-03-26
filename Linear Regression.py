import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(x, y):
    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def main():
    # Sample data
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 3, 4, 5, 6])

    slope, intercept = linear_regression(x, y)
    print("Slope:", slope)
    print("Intercept:", intercept)

if __name__ == "__main__":
    main()
