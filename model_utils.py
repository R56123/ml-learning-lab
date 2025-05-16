# Here I import the required Libraries for the ML.
import numpy as np # For mathematical operations (Arrays, dot product etc.)
import pandas as pd # For loading and working with csv files.
import matplotlib.pyplot as plt # For creating charts and plots.
from tqdm import tqdm # For showing the progress bar in the terminal.
from sklearn.preprocessing import StandardScaler # For scaling Input data.
import time # Used to add delays in training to visualise the progress bar. 

def load_data(filepath):
   # Automatically detect delimiter based on file type
    if "student" in filepath.lower():
        data = pd.read_csv(filepath)  # uses comma
    else:
        data = pd.read_csv(filepath, sep=';')  # wine dataset

    data.columns = data.columns.str.strip()  # clean up header spaces


    # ✅ Wine dataset logic
    if 'quality' in data.columns:
        X = data.drop('quality', axis=1).values
        y = data['quality'].values.reshape(-1, 1)

        # Standardize wine features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    # ✅ Student dataset logic
    elif 'Hours' in data.columns and 'Scores' in data.columns:
        X = data[['Hours']].values  # use only the Hours column
        y = data[['Scores']].values  # prediction target
        return X, y

    # ❌ Unsupported format
    else:
        raise ValueError("Unsupported dataset format. Expected columns: 'quality' or 'Hours' & 'Scores'.")


# Predict using weights and bias
def predict(X, weight, bias):
    # Return predicted data values using the formula: y = Xw + b
    return X.dot(weight) + bias

# Evaluate error between predictions and actual labels
def evaluate_model(y_true, y_pred):
    # Here I calculate the Mean Squared Error: An average of squared differences
    mse = np.mean((y_true - y_pred) ** 2)
    # Calculate the RMSE: square root of MSE
    rmse = np.sqrt(mse)
    # Return both error values
    return mse, rmse

# Plot regression line vs actual data
def plot_regression(X, y, y_pred):
    # Here I draw the actual scores via blue dots
    plt.scatter(X, y, color='blue', label='Actual Scores')
    # Here I draw the predicted line as a red line. 
    plt.plot(X, y_pred, color='red', label='Predicted Line')
    # Here I add labels and a title to the graph.
    plt.xlabel("Hours Studied")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.grid(True)
    # Display the graph window
    plt.show()

def train_model(X, y, learning_rate=0.001, epochs=100, label="Model", return_history=True):
    n_samples = X.shape[0] # Here I get the number of rows (samples)
    weight = np.random.randn(X.shape[1], 1) # Begin with random weights
    bias = 0 # Start with bias 0
    rmse_history = [] # A list to save RSME from each training round
    mse_history = [] # A list to save MSE from each training round

    progress = tqdm(range(1, epochs + 1), desc=f"{label} Training", colour='green', ncols=100) # Here I setup a clean progress bar.

    for epoch in progress:
        time.sleep(0.01)  # adds 50 milliseconds delay per epoch
        y_pred = predict(X, weight, bias) # Here I predict using current weights and bias
        dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred)) # Gradient for weights
        db = -(2 / n_samples) * np.sum(y - y_pred) # Gradient for bias.

        weight -= learning_rate * dw # Adjust weights 
        bias -= learning_rate * db # Adjust bias

        mse, rmse = evaluate_model(y, y_pred) # Here I evaluate the current performance.

        # I then store the current errors in the tracking history.
        mse_history.append(mse)
        rmse_history.append(rmse)

        # Here I print a detailed log in the terminal for every 10 epochs, plus the first and final epoch.
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            tqdm.write(f"{label} Epoch {epoch:3d}/{epochs} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
        # Here I simply update the progress bar label for every 5 epochs with the current RSME
        if epoch % 5 == 0:
            progress.set_description(f"{label} Epoch {epoch:3d} | RMSE: {rmse:.2f}")
        # Here I then return the final trained weights, bias and history of performance.
    if return_history:
        return weight, bias, rmse_history, mse_history
    else: 
        return weight, bias