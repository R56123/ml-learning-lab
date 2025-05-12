from model_utils import load_data, train_model, predict, evaluate_model, plot_regression

# Step 1: Load the dataset
X, y = load_data("data/student_scores.csv")

# Step 2: Train the model using gradient descent
weight, bias = train_model(X, y, learning_rate=0.03, epochs=100)

# Step 3: Make predictions
y_pred = predict(X, weight, bias)

# Step 4: Evaluate model performance
mse, rmse = evaluate_model(y, y_pred)
print(f"\n✅ Final Evaluation:\nMSE = {mse:.2f}, RMSE = {rmse:.2f}")

# Step 5: Visualize actual vs predicted scores
plot_regression(X, y, y_pred)
