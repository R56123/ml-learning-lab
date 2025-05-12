
# Here I import the necessary functions from the utility model.
from model_utils import load_data, train_model, predict, evaluate_model, plot_regression

# Step 1: Load the dataset for student scores. I return Input features (X) and labels (y).
X, y = load_data("data/student_scores.csv")

# Step 2: Train the model using gradient descent
# Here I pass the data to the training function, which returns the learned weight and bias. 
# Learning_rate controls how big each step is while training
# epochs defines how many times the model sees the full dataset
weight, bias = train_model(X, y, learning_rate=0.03, epochs=100)

# Step 3: Make predictions
# I then use the trained weight and bias to predict the scores based on hours studied
y_pred = predict(X, weight, bias)

# Step 4: Evaluate model performance
# Here I calculate how good the predictions are using MSE (Mean Squared Error) and RMSE (Root Mean Squared Error)
mse, rmse = evaluate_model(y, y_pred)
print(f"\n✅ Final Evaluation:\nMSE = {mse:.2f}, RMSE = {rmse:.2f}")

# Step 5: Visualize actual vs predicted scores
# Here I scatter plot of actual data and the predicted regression line. 
plot_regression(X, y, y_pred)
