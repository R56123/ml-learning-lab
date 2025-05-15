import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from model_utils import load_data, train_model, predict, evaluate_model, plot_regression

# Load Dataset
X, y = load_data("data/student_scores.csv")

# Use a high learning rate 
learning_rate = 0.1
epochs = 1000

print(f"\n Experiment 1: High Learning Rate = {learning_rate}\n")
weight, bias = train_model(X, y, learning_rate=learning_rate, epochs=epochs, return_history=False)

# Predictions 

y_pred = predict(X, weight, bias)

# Evaluate performance 

mse, rmse = evaluate_model(y, y_pred)
print(f"\n Final Evaluation:\nMSE = {mse:.2f}, RSME = {rmse:.2f}")

# Visualise
plot_regression(X, y, y_pred)