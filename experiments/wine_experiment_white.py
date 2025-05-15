import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from model_utils import load_data, train_model, predict, evaluate_model

# Load Wine Data
X, y = load_data("data/winequality-white.csv")

# Train
learning_rate = 0.001
epochs = 100
weight, bias, rmse_history, mse_history = train_model(X, y, learning_rate=learning_rate, epochs=epochs)

# Evaluate
y_pred = predict(X, weight, bias)
mse, rmse = evaluate_model(y, y_pred)
print(f"\nWhite Wine Quality Prediction:\nMSE = {mse:.2f}, RMSE = {rmse:.2f}")
