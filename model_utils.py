import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import time

def load_data(filepath):
    data = pd.read_csv(filepath, sep=';')
    X = data.drop('quality', axis=1).values
    y = data['quality'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Predict using weights and bias
def predict(X, weight, bias):
    return X.dot(weight) + bias

# Evaluate error between predictions and actual labels
def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse

# Plot regression line vs actual data
def plot_regression(X, y, y_pred):
    plt.scatter(X, y, color='blue', label='Actual Scores')
    plt.plot(X, y_pred, color='red', label='Predicted Line')
    plt.xlabel("Hours Studied")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.grid(True)
    plt.show()

def train_model(X, y, learning_rate=0.001, epochs=100, label="Model"):
    n_samples = X.shape[0]
    weight = np.random.randn(X.shape[1], 1)
    bias = 0
    rmse_history = []
    mse_history = []

    progress = tqdm(range(1, epochs + 1), desc=f"{label} Training", colour='green', ncols=100)

    for epoch in progress:
        time.sleep(0.05)  # adds 50 milliseconds delay per epoch
        y_pred = predict(X, weight, bias)
        dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
        db = -(2 / n_samples) * np.sum(y - y_pred)

        weight -= learning_rate * dw
        bias -= learning_rate * db

        mse, rmse = evaluate_model(y, y_pred)
        mse_history.append(mse)
        rmse_history.append(rmse)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            tqdm.write(f"{label} Epoch {epoch:3d}/{epochs} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")

        if epoch % 5 == 0:
            progress.set_description(f"{label} Epoch {epoch:3d} | RMSE: {rmse:.2f}")

    return weight, bias, rmse_history, mse_history
